"""
Anchor Quorum & Dual-Anchor Independence Architecture.
Implements Section 5 configuration, independence verification, and quorum policies.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Set, Final

from analyst_dashboard.smart_money.anchor_finalization.models import (
    QuorumPolicy,
    QuorumNotSatisfiedError,
)

# ---------------------------------------------------------------------------
# Default Configured Providers (Section 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnchorProviderConfig:
    provider_id: str
    credential_domain: str
    key_domain: str
    admin_domain: str
    write_authority: str


DEFAULT_PROVIDER_A: Final[AnchorProviderConfig] = AnchorProviderConfig(
    provider_id="AWS_S3_OBJECT_LOCK",
    credential_domain="AWS_IAM_ROLE_ARX_ANCHOR_A",
    key_domain="AWS_KMS_KEY_ARX_ANCHOR_A",
    admin_domain="AWS_ACCOUNT_SECURITY_ORG_A",
    write_authority="PUT_OBJECT_LEGAL_HOLD_WORM",
)

DEFAULT_PROVIDER_B: Final[AnchorProviderConfig] = AnchorProviderConfig(
    provider_id="GCP_CLOUD_STORAGE_LOCK",
    credential_domain="GCP_SERVICE_ACCOUNT_ARX_ANCHOR_B",
    key_domain="GCP_KMS_KEY_ARX_ANCHOR_B",
    admin_domain="GCP_PROJECT_SECURITY_ORG_B",
    write_authority="INSERT_OBJECT_RETENTION_LOCK",
)

ANCHOR_PROVIDER_COUNT: Final[int] = 2
ANCHOR_REQUIRED_COUNT: Final[int] = 2
ANCHOR_QUORUM_POLICY: Final[QuorumPolicy] = QuorumPolicy.ALL_REQUIRED


def verify_dual_anchor_independence(
    provider_a: AnchorProviderConfig,
    provider_b: AnchorProviderConfig,
) -> str:
    """
    Verifies dual-anchor independence (Section 5).
    Required for claiming independence:
    - separate credential domain
    - separate write authority
    - separate key domain
    - separate admin domain
    Returns: 'PASS' if independent, 'FAIL' otherwise.
    """
    if provider_a.credential_domain == provider_b.credential_domain:
        return "FAIL"
    if provider_a.write_authority == provider_b.write_authority:
        return "FAIL"
    if provider_a.key_domain == provider_b.key_domain:
        return "FAIL"
    if provider_a.admin_domain == provider_b.admin_domain:
        return "FAIL"
    if provider_a.provider_id == provider_b.provider_id:
        return "FAIL"
    return "PASS"


def evaluate_anchor_quorum(
    confirmed_providers: List[str],
    required_providers: List[str] = None,
    quorum_policy: QuorumPolicy = QuorumPolicy.ALL_REQUIRED,
) -> bool:
    """
    Evaluates whether confirmed providers satisfy the quorum policy (Section 5).
    Invariants:
    - Does not infer quorum dynamically.
    - Outages CANNOT lower quorum.
    """
    if required_providers is None:
        required_providers = [DEFAULT_PROVIDER_A.provider_id, DEFAULT_PROVIDER_B.provider_id]

    confirmed_set = set(confirmed_providers)
    required_set = set(required_providers)

    if quorum_policy == QuorumPolicy.ALL_REQUIRED:
        return required_set.issubset(confirmed_set)
    elif quorum_policy == QuorumPolicy.ONE_OF_TWO:
        return len(confirmed_set.intersection(required_set)) >= 1
    elif quorum_policy == QuorumPolicy.TWO_OF_THREE:
        return len(confirmed_set.intersection(required_set)) >= 2
    else:
        raise ValueError(f"Unknown quorum policy: {quorum_policy}")


def assert_anchor_quorum_satisfied(
    confirmed_providers: List[str],
    required_providers: List[str] = None,
    quorum_policy: QuorumPolicy = QuorumPolicy.ALL_REQUIRED,
) -> None:
    """Enforces quorum satisfaction or raises QuorumNotSatisfiedError."""
    if not evaluate_anchor_quorum(confirmed_providers, required_providers, quorum_policy):
        raise QuorumNotSatisfiedError(
            f"QUORUM_NOT_SATISFIED: Confirmed providers {confirmed_providers} "
            f"do not satisfy policy {quorum_policy} for required {required_providers}"
        )
