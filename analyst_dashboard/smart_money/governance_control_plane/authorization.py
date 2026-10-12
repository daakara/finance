"""
Authorization Subsystem
ARX Terminal — Smart Money

Implements strict Decision != Authorization separation, TOCTOU protection,
operator identity binding, least-authority action scopes, single-use tracking, and TTL.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import uuid
from datetime import datetime, timezone, timedelta

from .canonicalization import sha256_canonical
from .ledger import AuthorityLedger

if TYPE_CHECKING:
    from .decision_engine import DecisionSnapshot


class ProvisioningBlockedException(Exception):
    """Raised when attempting to issue authorization while decision readiness is BLOCKED."""
    pass


class StaleDecisionException(Exception):
    """Raised when decision snapshot has been superseded by newer authority head."""
    pass


class TOCTOUException(Exception):
    """Raised when authority head or registry changed between authorization and execution."""
    pass


class WrongOperatorException(Exception):
    """Raised when caller identity does not match authorized operator identity."""
    pass


class OutOfScopeActionException(Exception):
    """Raised when requested action is not within authorization's allowed_actions."""
    pass


class AuthorizationExpiredException(Exception):
    """Raised when authorization is exercised after expires_at."""
    pass


class AuthorizationConsumedException(Exception):
    """Raised when single-use authorization has already been exercised."""
    pass


@dataclass
class ProvisioningAuthorization:
    authorization_id: str
    authorization_type: str
    governance_epoch: str
    decision_hash: str
    policy_hash: str
    ledger_head_hash: str
    registry_hash: str
    operator_identity: str
    allowed_actions: List[str]
    resource_scope: List[str]
    theorem_scope: List[str]
    issued_sequence: int
    issued_at: str
    expires_at: str
    single_use: bool
    max_uses: int
    precondition_hashes: Dict[str, str]
    authorization_hash: str
    signature_or_ratification_ref: str
    evidence_manifest_hash: Optional[str] = None
    dependency_resolution_hash: Optional[str] = None
    consumed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("consumed_count", None)
        return d


@dataclass
class ExecutionAuthorization:
    authorization_id: str
    authorization_type: str
    governance_epoch: str
    decision_hash: str
    policy_hash: str
    ledger_head_hash: str
    registry_hash: str
    operator_identity: str
    allowed_actions: List[str]
    resource_scope: List[str]
    theorem_scope: List[str]
    issued_sequence: int
    issued_at: str
    expires_at: str
    single_use: bool
    max_uses: int
    precondition_hashes: Dict[str, str]
    authorization_hash: str
    signature_or_ratification_ref: str
    evidence_manifest_hash: Optional[str] = None
    dependency_resolution_hash: Optional[str] = None
    consumed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("consumed_count", None)
        return d


class AuthorizationManager:
    """
    Issues, validates, and consumes time-bound, actor-bound, scope-bound authorizations.
    Enforces that Decision != Authorization and prevents TOCTOU exploits.
    """

    PROVISIONING_TTL_SECONDS = 3600
    EXECUTION_TTL_SECONDS = 900

    def __init__(self, ledger: AuthorityLedger):
        self.ledger = ledger
        self.active_authorizations: Dict[str, Any] = {}

    def issue_provisioning_authorization(
        self,
        decision: DecisionSnapshot,
        operator_identity: str,
        allowed_actions: List[str],
        resource_scope: List[str],
        theorem_scope: List[str],
        current_registry_hash: str,
        current_evidence_manifest_hash: Optional[str] = None,
        current_dependency_resolution_hash: Optional[str] = None,
        issued_at: Optional[str] = None,
        ttl_seconds: int = PROVISIONING_TTL_SECONDS,
        require_dependency_binding: bool = True,
    ) -> ProvisioningAuthorization:
        # 1. Decision must be READY
        if decision.global_provisioning_readiness != "READY":
            raise ProvisioningBlockedException(
                f"Cannot issue provisioning authorization: readiness is {decision.global_provisioning_readiness}."
            )

        # Canonical eligibility check
        if getattr(decision, "canonical_eligibility", "PASS") != "PASS" or getattr(decision, "authority_eligibility", "AUTHORITATIVE") != "AUTHORITATIVE":
            raise ProvisioningBlockedException(
                "Cannot issue authorization: decision has synthetic or non-authoritative evidence."
            )

        # Resolve evidence and dependency hashes
        ev_hash = current_evidence_manifest_hash or getattr(decision, "evidence_manifest_hash", None)
        dep_hash = current_dependency_resolution_hash or getattr(decision, "dependency_resolution_hash", None)

        if require_dependency_binding:
            if not dep_hash:
                raise ValueError("Authorization requires dependency_resolution_hash binding (AUTHORIZATION_WITHOUT_DEPENDENCY_BINDING_ACCEPTED == 0).")
            if not ev_hash:
                raise ValueError("Authorization requires evidence_manifest_hash binding (AUTHORIZATION_WITHOUT_EVIDENCE_BINDING_ACCEPTED == 0).")
            if not current_registry_hash or not getattr(decision, "finding_registry_hash", None):
                raise ValueError("Authorization requires finding_registry_hash binding (AUTHORIZATION_WITHOUT_FINDING_BINDING_ACCEPTED == 0).")

        # 2. Decision must not be stale
        if decision.finding_registry_hash != current_registry_hash:
            raise StaleDecisionException(
                "Cannot issue authorization: decision registry hash does not match current registry hash."
            )

        if current_evidence_manifest_hash and getattr(decision, "evidence_manifest_hash", None):
            if current_evidence_manifest_hash != decision.evidence_manifest_hash:
                raise StaleDecisionException(
                    "Cannot issue authorization: decision evidence manifest hash does not match current evidence manifest hash."
                )

        if current_dependency_resolution_hash and getattr(decision, "dependency_resolution_hash", None):
            if current_dependency_resolution_hash != decision.dependency_resolution_hash:
                raise StaleDecisionException(
                    "Cannot issue authorization: decision dependency resolution hash does not match current dependency resolution hash."
                )

        now = datetime.now(timezone.utc)
        issued_at_str = issued_at or now.strftime("%Y-%m-%dT%H:%M:%SZ")
        expires_at_dt = now + timedelta(seconds=ttl_seconds)
        expires_at_str = expires_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        auth_id = f"urn:arx:auth:prov:{decision.governance_epoch}:{uuid.uuid4().hex[:12]}"
        head_hash = self.ledger.head_hash or ("0" * 64)

        body = {
            "authorization_id": auth_id,
            "authorization_type": "PROVISIONING",
            "governance_epoch": decision.governance_epoch,
            "decision_hash": decision.decision_hash,
            "policy_hash": decision.policy_hash,
            "ledger_head_hash": head_hash,
            "registry_hash": current_registry_hash,
            "operator_identity": operator_identity,
            "allowed_actions": sorted(allowed_actions),
            "resource_scope": sorted(resource_scope),
            "theorem_scope": sorted(theorem_scope),
            "issued_sequence": self.ledger.sequence,
            "issued_at": issued_at_str,
            "expires_at": expires_at_str,
            "single_use": True,
            "max_uses": 1,
            "precondition_hashes": {
                "decision_hash": decision.decision_hash,
                "policy_hash": decision.policy_hash,
                "evidence_manifest_hash": ev_hash or "",
                "dependency_resolution_hash": dep_hash or "",
                "registry_hash": current_registry_hash,
            },
            "signature_or_ratification_ref": "urn:arx:ratification:council:prov-auth-001",
        }
        auth_hash = sha256_canonical(body)

        auth = ProvisioningAuthorization(
            authorization_id=auth_id,
            authorization_type="PROVISIONING",
            governance_epoch=decision.governance_epoch,
            decision_hash=decision.decision_hash,
            policy_hash=decision.policy_hash,
            ledger_head_hash=head_hash,
            registry_hash=current_registry_hash,
            operator_identity=operator_identity,
            allowed_actions=allowed_actions,
            resource_scope=resource_scope,
            theorem_scope=theorem_scope,
            issued_sequence=self.ledger.sequence,
            issued_at=issued_at_str,
            expires_at=expires_at_str,
            single_use=True,
            max_uses=1,
            precondition_hashes=body["precondition_hashes"],
            authorization_hash=auth_hash,
            signature_or_ratification_ref=body["signature_or_ratification_ref"],
            evidence_manifest_hash=ev_hash,
            dependency_resolution_hash=dep_hash,
        )
        self.active_authorizations[auth_id] = auth
        return auth

    def validate_and_consume(
        self,
        auth_id: str,
        caller_identity: str,
        requested_action: str,
        current_decision_hash: str,
        current_registry_hash: str,
        current_evidence_manifest_hash: Optional[str] = None,
        current_dependency_resolution_hash: Optional[str] = None,
        check_time: Optional[datetime] = None,
    ) -> None:
        if auth_id not in self.active_authorizations:
            raise OutOfScopeActionException(f"Authorization {auth_id} not found.")

        auth = self.active_authorizations[auth_id]

        # 1. Check consumption
        if auth.consumed_count >= auth.max_uses:
            raise AuthorizationConsumedException(
                f"Authorization {auth_id} already consumed ({auth.consumed_count}/{auth.max_uses})."
            )

        # 2. Check Expiry
        now = check_time or datetime.now(timezone.utc)
        exp_dt = datetime.fromisoformat(auth.expires_at.replace("Z", "+00:00"))
        if now > exp_dt:
            raise AuthorizationExpiredException(
                f"Authorization {auth_id} expired at {auth.expires_at} (current time {now.isoformat()})."
            )

        # 3. Check Operator Identity
        if caller_identity != auth.operator_identity:
            raise WrongOperatorException(
                f"Caller identity {caller_identity} does not match authorized operator {auth.operator_identity}."
            )

        # 4. Check Action Scope
        if requested_action not in auth.allowed_actions:
            raise OutOfScopeActionException(
                f"Action {requested_action} is outside permitted actions {auth.allowed_actions}."
            )

        # 5. TOCTOU Protection: Authority Head Check
        current_head = self.ledger.head_hash or ("0" * 64)
        if current_head != auth.ledger_head_hash:
            raise TOCTOUException(
                f"TOCTOU check failed: current ledger head {current_head} "
                f"does not match authorization ledger head {auth.ledger_head_hash}."
            )

        if current_decision_hash != auth.decision_hash:
            raise TOCTOUException(
                f"TOCTOU check failed: current decision hash {current_decision_hash} "
                f"does not match authorization decision hash {auth.decision_hash}."
            )

        if current_registry_hash != auth.registry_hash:
            raise TOCTOUException(
                f"TOCTOU check failed: current registry hash {current_registry_hash} "
                f"does not match authorization registry hash {auth.registry_hash}."
            )

        if auth.evidence_manifest_hash and current_evidence_manifest_hash:
            if current_evidence_manifest_hash != auth.evidence_manifest_hash:
                raise TOCTOUException(
                    f"TOCTOU check failed: current evidence manifest hash {current_evidence_manifest_hash} "
                    f"does not match authorization evidence manifest hash {auth.evidence_manifest_hash}. "
                    f"STALE_EVIDENCE_AUTHORIZATION_ACCEPTED == 0."
                )

        if auth.dependency_resolution_hash and current_dependency_resolution_hash:
            if current_dependency_resolution_hash != auth.dependency_resolution_hash:
                raise TOCTOUException(
                    f"TOCTOU check failed: current dependency resolution hash {current_dependency_resolution_hash} "
                    f"does not match authorization dependency resolution hash {auth.dependency_resolution_hash}. "
                    f"STALE_DEPENDENCY_AUTHORIZATION_ACCEPTED == 0."
                )

        # Consume
        auth.consumed_count += 1
