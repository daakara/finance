"""
External Anchor Storage Configuration Attestation (Section 16).
Captures and attests configuration evidence for WORM / Object Lock providers.
"""

import hashlib
import json
from typing import Dict, Any, Final

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.models import (
    StorageConfigurationAttestation,
)

CANONICAL_WORM_RETENTION_UNTIL: Final[str] = "2036-10-11T12:00:00Z"


def create_storage_configuration_attestation(
    provider: str,
    object_lock_enabled: bool = True,
    versioning_enabled: bool = True,
    retention_mode: str = "COMPLIANCE",
    retention_until: str = CANONICAL_WORM_RETENTION_UNTIL,
    delete_protection: bool = True,
    overwrite_protection: bool = True,
    config_captured_at: str = "2026-10-11T12:00:00Z",
) -> StorageConfigurationAttestation:
    """
    Creates an attested storage configuration record with SHA256 evidence.
    """
    evidence_dict = {
        "config_captured_at": config_captured_at,
        "delete_protection": delete_protection,
        "object_lock_enabled": object_lock_enabled,
        "overwrite_protection": overwrite_protection,
        "provider": provider,
        "retention_mode": retention_mode,
        "retention_until": retention_until,
        "versioning_enabled": versioning_enabled,
    }
    canon_bytes = canonicalize_json(evidence_dict).encode("utf-8")
    evidence_sha = hashlib.sha256(canon_bytes).hexdigest()

    return StorageConfigurationAttestation(
        provider=provider,
        object_lock_enabled=object_lock_enabled,
        versioning_enabled=versioning_enabled,
        retention_mode=retention_mode,
        retention_until=retention_until,
        delete_protection=delete_protection,
        overwrite_protection=overwrite_protection,
        config_captured_at=config_captured_at,
        config_evidence_sha256=evidence_sha,
        immutability_class="PROVIDER_ENFORCED_RETENTION",
        trust_assumption="PROVIDER_CONTROL_PLANE_NOT_COMPROMISED",
    )


def verify_storage_configuration_attestation(
    attestation: StorageConfigurationAttestation,
) -> bool:
    """
    Verifies that the attestation satisfies minimum WORM / Object Lock requirements (Section 16).
    """
    if not attestation.object_lock_enabled:
        return False
    if not attestation.versioning_enabled:
        return False
    if attestation.retention_mode not in ("COMPLIANCE", "GOVERNANCE"):
        return False
    if not attestation.delete_protection:
        return False
    if not attestation.overwrite_protection:
        return False
    if attestation.immutability_class != "PROVIDER_ENFORCED_RETENTION":
        return False
    if attestation.trust_assumption != "PROVIDER_CONTROL_PLANE_NOT_COMPROMISED":
        return False
    return True
