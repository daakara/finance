"""
Canonical Policy Registry, Epoch Registry & Deterministic Policy Resolution Architecture.
Implements Sections 4..18 of Anchor Policy Resolution Closure Gate:
- Canonical Epoch Registry (string -> integer ordinal)
- Versioned Policy Registry (half-open [valid_from, valid_until) intervals)
- Run-Start Time Interval Authority
- Boundary Classification (BEFORE, AFTER_OR_EQUAL, STRADDLES_BOUNDARY)
- Fail-Closed Straddle Rule (INDETERMINATE_BOUNDARY)
- Exactly-One Policy Selection & Conflict Detection
- Normal Supersession & Emergency Revocation Semantics
- Run Genesis Immutable Policy Binding
- Independent Replay Contract
"""

import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Final

from analyst_dashboard.smart_money.canonicalization import canonicalize_json

def _parse_utc_iso(ts_str: str) -> datetime:
    normalized = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RevocationStatus(str, Enum):
    """Emergency and administrative revocation statuses (Section 14)."""
    ACTIVE = "ACTIVE"
    REVOKED_FOR_FUTURE_RUNS = "REVOKED_FOR_FUTURE_RUNS"
    REVOKED_FOR_AUTHORITY_USE = "REVOKED_FOR_AUTHORITY_USE"


class BoundaryClassification(str, Enum):
    """Clock boundary relation classification (Section 9)."""
    BEFORE = "BEFORE"
    AFTER_OR_EQUAL = "AFTER_OR_EQUAL"
    STRADDLES_BOUNDARY = "STRADDLES_BOUNDARY"
    AFTER_POLICY = "AFTER_POLICY"
    BEFORE_POLICY_END = "BEFORE_POLICY_END"
    STRADDLES_POLICY_END = "STRADDLES_POLICY_END"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PolicyResolutionResult(str, Enum):
    """Outcomes of deterministic policy resolution (Sections 10, 11, 12)."""
    APPLICABLE = "APPLICABLE"
    INDETERMINATE_BOUNDARY = "INDETERMINATE_BOUNDARY"
    NO_APPLICABLE_POLICY = "NO_APPLICABLE_POLICY"
    CONFLICTING_APPLICABLE_POLICIES = "CONFLICTING_APPLICABLE_POLICIES"
    BLOCKED_UNKNOWN_EPOCH = "BLOCKED_UNKNOWN_EPOCH"
    BLOCKED_REVOKED_POLICY = "BLOCKED_REVOKED_POLICY"
    BLOCKED_NO_POLICY = "BLOCKED_NO_POLICY"


class PreEffectivePolicyBehavior(str, Enum):
    """Behavior when a run starts prior to policy activation (Section 12)."""
    PRIOR_POLICY = "PRIOR_POLICY"
    NO_ANCHOR_REQUIREMENT = "NO_ANCHOR_REQUIREMENT"
    BLOCKED_NO_POLICY = "BLOCKED_NO_POLICY"


# ---------------------------------------------------------------------------
# Epoch Registry (Section 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpochRegistry:
    """
    Authoritative epoch registry mapping symbolic names to integer ordinals.
    Enforces that epoch order is never inferred from strings.
    """
    version: str = "1.0"
    mapping: Dict[str, int] = field(default_factory=lambda: {
        "EPOCH_001": 1,
        "EPOCH_002": 2,
        "EPOCH_003": 3,
        "EPOCH_004": 4,
    })

    def get_ordinal(self, epoch_name: str) -> Optional[int]:
        return self.mapping.get(epoch_name)

    def canonical_bytes(self) -> bytes:
        data = {
            "version": self.version,
            "mapping": self.mapping,
        }
        return canonicalize_json(data).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


DEFAULT_EPOCH_REGISTRY: Final[EpochRegistry] = EpochRegistry()
EPOCH_REGISTRY_VERSION: Final[str] = DEFAULT_EPOCH_REGISTRY.version
EPOCH_REGISTRY_SHA256: Final[str] = DEFAULT_EPOCH_REGISTRY.sha256()


# ---------------------------------------------------------------------------
# Policy Registry Entry & Registry (Section 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyRegistryEntry:
    """
    Versioned anchor policy registry entry under half-open [valid_from, valid_until).
    """
    policy_id: str
    policy_version: str
    policy_sha256: str
    valid_from: str  # ISO 8601 UTC
    valid_until: str  # ISO 8601 UTC or "OPEN"
    min_run_epoch_ordinal: int
    revocation_status: RevocationStatus = RevocationStatus.ACTIVE
    revocation_effective_at: Optional[str] = None
    revocation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_run_epoch_ordinal": self.min_run_epoch_ordinal,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "policy_version": self.policy_version,
            "revocation_effective_at": self.revocation_effective_at,
            "revocation_reason": self.revocation_reason,
            "revocation_status": self.revocation_status.value,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
        }


@dataclass(frozen=True)
class PolicyRegistry:
    """
    Authoritative append-only policy registry.
    """
    version: str = "1.0"
    entries: List[PolicyRegistryEntry] = field(default_factory=list)

    def canonical_bytes(self) -> bytes:
        serialized_entries = [e.to_dict() for e in self.entries]
        data = {
            "version": self.version,
            "entries": serialized_entries,
        }
        return canonicalize_json(data).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate_integrity(self) -> None:
        """
        Validates internal consistency: no duplicate IDs, ordered entries,
        no interval overlaps for same policy lineage.
        """
        seen_keys = set()
        for i, entry in enumerate(self.entries):
            key = (entry.policy_id, entry.policy_version)
            if key in seen_keys:
                raise ValueError(f"Duplicate policy entry: {key}")
            seen_keys.add(key)

            # Check overlap against other entries of same policy family
            for j in range(i + 1, len(self.entries)):
                other = self.entries[j]
                if other.policy_id == entry.policy_id:
                    # Check overlap on [valid_from, valid_until)
                    e_from = _parse_utc_iso(entry.valid_from)
                    e_until = _parse_utc_iso(entry.valid_until) if entry.valid_until != "OPEN" else None
                    o_from = _parse_utc_iso(other.valid_from)
                    o_until = _parse_utc_iso(other.valid_until) if other.valid_until != "OPEN" else None

                    overlap = True
                    if e_until is not None and o_from >= e_until:
                        overlap = False
                    if o_until is not None and e_from >= o_until:
                        overlap = False
                    if overlap:
                        raise ValueError(f"Overlapping validity intervals between {key} and {(other.policy_id, other.policy_version)}")


# Canonical V1 Registry Definition
CANONICAL_ANCHOR_POLICY_V1_ENTRY: Final[PolicyRegistryEntry] = PolicyRegistryEntry(
    policy_id="SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1",
    policy_version="1.0",
    policy_sha256="f5926ecdb5536443c5b5502c4bce81452a382e7e43d4107b3ecf6d92ec3a4176",
    valid_from="2026-10-11T12:00:00Z",
    valid_until="OPEN",
    min_run_epoch_ordinal=1,
    revocation_status=RevocationStatus.ACTIVE,
    revocation_effective_at=None,
    revocation_reason=None,
)

DEFAULT_POLICY_REGISTRY: Final[PolicyRegistry] = PolicyRegistry(
    version="1.0",
    entries=[CANONICAL_ANCHOR_POLICY_V1_ENTRY],
)
POLICY_REGISTRY_VERSION: Final[str] = DEFAULT_POLICY_REGISTRY.version
POLICY_REGISTRY_SHA256: Final[str] = DEFAULT_POLICY_REGISTRY.sha256()


# ---------------------------------------------------------------------------
# Run-Start Time Interval & Binding Models (Sections 8, 15)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunStartInterval:
    """
    Authoritative run-start interval input (Section 8).
    Interval bounds are derived from clock evidence (observed time +/- uncertainty).
    """
    run_genesis_event_id: str
    run_genesis_sequence: int
    run_started_at_local: str
    run_start_clock_evidence_id: str
    run_started_at_interval_start: str  # T_start
    run_started_at_interval_end: str    # T_end
    run_epoch: str


@dataclass(frozen=True)
class RunPolicyBinding:
    """
    Immutable policy binding record sealed at run genesis (Section 15).
    """
    run_policy_binding_id: str
    bound_policy_id: Optional[str]
    bound_policy_version: Optional[str]
    bound_policy_sha256: Optional[str]
    bound_policy_registry_sha256: str
    bound_epoch_registry_sha256: str
    run_epoch: str
    run_epoch_ordinal: int
    run_start_interval_start: str
    run_start_interval_end: str
    policy_applicability_result: PolicyResolutionResult
    boundary_classification: BoundaryClassification
    policy_binding_at: str
    policy_binding_event_id: str
    is_immutable: bool = True


# ---------------------------------------------------------------------------
# Boundary Classification Logic (Section 9)
# ---------------------------------------------------------------------------
def classify_run_interval_boundary(
    interval_start: str,
    interval_end: str,
    target_boundary: str,
) -> BoundaryClassification:
    """
    Classifies run-start interval against a point boundary (valid_from):
    - if interval_end < target_boundary -> BEFORE
    - if interval_start >= target_boundary -> AFTER_OR_EQUAL
    - otherwise -> STRADDLES_BOUNDARY
    """
    t_start = _parse_utc_iso(interval_start)
    t_end = _parse_utc_iso(interval_end)
    t_bound = _parse_utc_iso(target_boundary)

    if t_end < t_bound:
        return BoundaryClassification.BEFORE
    if t_start >= t_bound:
        return BoundaryClassification.AFTER_OR_EQUAL
    return BoundaryClassification.STRADDLES_BOUNDARY


def classify_run_interval_policy_end(
    interval_start: str,
    interval_end: str,
    valid_until: str,
) -> BoundaryClassification:
    """
    Classifies run-start interval against a policy termination boundary (valid_until):
    - if interval_start >= valid_until -> AFTER_POLICY
    - if interval_end < valid_until -> BEFORE_POLICY_END
    - otherwise -> STRADDLES_POLICY_END
    """
    if valid_until == "OPEN":
        return BoundaryClassification.BEFORE_POLICY_END
    t_start = _parse_utc_iso(interval_start)
    t_end = _parse_utc_iso(interval_end)
    t_until = _parse_utc_iso(valid_until)

    if t_start >= t_until:
        return BoundaryClassification.AFTER_POLICY
    if t_end < t_until:
        return BoundaryClassification.BEFORE_POLICY_END
    return BoundaryClassification.STRADDLES_POLICY_END


# ---------------------------------------------------------------------------
# Policy Resolution Engine (Sections 7, 10, 11, 12, 13, 14)
# ---------------------------------------------------------------------------
class PolicyResolutionEngine:
    """
    Deterministic Policy Applicability Resolver.
    Enforces ALL_PREDICATES_REQUIRED, fail-closed straddles, exactly-one selection,
    and handles normal supersession and emergency revocation.
    """

    def __init__(
        self,
        policy_registry: Optional[PolicyRegistry] = None,
        epoch_registry: Optional[EpochRegistry] = None,
        prior_policy_id: Optional[str] = None,
        pre_effective_behavior: PreEffectivePolicyBehavior = PreEffectivePolicyBehavior.BLOCKED_NO_POLICY,
    ):
        self.policy_registry = policy_registry or DEFAULT_POLICY_REGISTRY
        self.epoch_registry = epoch_registry or DEFAULT_EPOCH_REGISTRY
        self.prior_policy_id = prior_policy_id
        self.pre_effective_behavior = pre_effective_behavior

    def resolve_policy_for_run(
        self,
        interval: RunStartInterval,
        binding_time_utc: Optional[str] = None,
    ) -> Tuple[PolicyResolutionResult, Optional[PolicyRegistryEntry], BoundaryClassification]:
        """
        Resolves applicable policy for a run.
        Returns (result, selected_policy_entry, boundary_classification).
        """
        # 1. Resolve Epoch Ordinal (Section 5)
        epoch_ordinal = self.epoch_registry.get_ordinal(interval.run_epoch)
        if epoch_ordinal is None:
            return (
                PolicyResolutionResult.BLOCKED_UNKNOWN_EPOCH,
                None,
                BoundaryClassification.NOT_APPLICABLE,
            )

        candidate_policies: List[Tuple[PolicyRegistryEntry, BoundaryClassification]] = []
        has_straddle = False
        primary_boundary_class = BoundaryClassification.NOT_APPLICABLE

        # 2. Evaluate each policy in the registry
        for entry in self.policy_registry.entries:
            # Check Epoch Predicate
            if epoch_ordinal < entry.min_run_epoch_ordinal:
                continue

            # Classify valid_from boundary
            start_class = classify_run_interval_boundary(
                interval.run_started_at_interval_start,
                interval.run_started_at_interval_end,
                entry.valid_from,
            )
            primary_boundary_class = start_class

            if start_class == BoundaryClassification.STRADDLES_BOUNDARY:
                has_straddle = True
                continue

            if start_class != BoundaryClassification.AFTER_OR_EQUAL:
                # Run is before this policy's valid_from
                continue

            # Classify valid_until boundary
            end_class = classify_run_interval_policy_end(
                interval.run_started_at_interval_start,
                interval.run_started_at_interval_end,
                entry.valid_until,
            )
            if end_class == BoundaryClassification.STRADDLES_POLICY_END:
                has_straddle = True
                continue

            if end_class != BoundaryClassification.BEFORE_POLICY_END:
                # Run is after this policy's valid_until
                continue

            # Check Revocation Status at binding time
            if entry.revocation_status == RevocationStatus.REVOKED_FOR_AUTHORITY_USE:
                continue
            if entry.revocation_status == RevocationStatus.REVOKED_FOR_FUTURE_RUNS:
                continue

            candidate_policies.append((entry, start_class))

        # 3. Handle Straddles (Section 10)
        if has_straddle:
            return (
                PolicyResolutionResult.INDETERMINATE_BOUNDARY,
                None,
                BoundaryClassification.STRADDLES_BOUNDARY,
            )

        # 4. Exactly-One Policy Selection (Section 11)
        count = len(candidate_policies)
        if count == 1:
            selected_entry, b_class = candidate_policies[0]
            return PolicyResolutionResult.APPLICABLE, selected_entry, b_class

        elif count > 1:
            return (
                PolicyResolutionResult.CONFLICTING_APPLICABLE_POLICIES,
                None,
                BoundaryClassification.NOT_APPLICABLE,
            )

        else:
            # 0 policies candidate-applicable (Section 12)
            if self.prior_policy_id is not None:
                # Prior policy lookup
                for p in self.policy_registry.entries:
                    if p.policy_id == self.prior_policy_id:
                        return PolicyResolutionResult.APPLICABLE, p, BoundaryClassification.BEFORE
            return (
                PolicyResolutionResult.NO_APPLICABLE_POLICY,
                None,
                primary_boundary_class,
            )

    def bind_run_genesis(
        self,
        interval: RunStartInterval,
        binding_event_id: str,
        binding_time_utc: str,
    ) -> RunPolicyBinding:
        """
        Creates an immutable policy binding at run genesis (Section 15).
        """
        result, policy_entry, b_class = self.resolve_policy_for_run(
            interval, binding_time_utc=binding_time_utc
        )

        epoch_ordinal = self.epoch_registry.get_ordinal(interval.run_epoch) or 0
        binding_id = f"bind_{interval.run_genesis_event_id}_{interval.run_epoch}"

        return RunPolicyBinding(
            run_policy_binding_id=binding_id,
            bound_policy_id=policy_entry.policy_id if policy_entry else None,
            bound_policy_version=policy_entry.policy_version if policy_entry else None,
            bound_policy_sha256=policy_entry.policy_sha256 if policy_entry else None,
            bound_policy_registry_sha256=self.policy_registry.sha256(),
            bound_epoch_registry_sha256=self.epoch_registry.sha256(),
            run_epoch=interval.run_epoch,
            run_epoch_ordinal=epoch_ordinal,
            run_start_interval_start=interval.run_started_at_interval_start,
            run_start_interval_end=interval.run_started_at_interval_end,
            policy_applicability_result=result,
            boundary_classification=b_class,
            policy_binding_at=binding_time_utc,
            policy_binding_event_id=binding_event_id,
            is_immutable=True,
        )


# ---------------------------------------------------------------------------
# Policy Replay Verifier (Sections 17, 18)
# ---------------------------------------------------------------------------
def replay_policy_binding(
    raw_genesis_metadata: Dict[str, Any],
    interval: RunStartInterval,
    historical_policy_registry: PolicyRegistry,
    historical_epoch_registry: EpochRegistry,
    prior_policy_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Independent historical replay of policy binding (Section 17).
    Uses ONLY raw genesis metadata, run-start interval, and historical registries.
    Never consumes stored binding decisions.
    """
    engine = PolicyResolutionEngine(
        policy_registry=historical_policy_registry,
        epoch_registry=historical_epoch_registry,
        prior_policy_id=prior_policy_id,
    )
    result, entry, b_class = engine.resolve_policy_for_run(interval)
    epoch_ordinal = historical_epoch_registry.get_ordinal(interval.run_epoch) or 0

    return {
        "replayed_run_epoch_ordinal": epoch_ordinal,
        "replayed_boundary_classification": b_class.value,
        "replayed_policy_applicability_result": result.value,
        "replayed_applicable_policy_count": 1 if entry else 0,
        "replayed_bound_policy_id": entry.policy_id if entry else None,
        "replayed_bound_policy_version": entry.policy_version if entry else None,
        "replayed_bound_policy_sha256": entry.policy_sha256 if entry else None,
    }
