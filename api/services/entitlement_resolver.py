"""
Entitlement contracts and default resolver for ARX SaaS Foundation.

This module defines:
1. EntitlementSet: Immutable collection supporting Boolean capability lookup
   and integer limit lookup.
2. EntitlementResolver: Protocol defining capability and limit resolution.
3. DefaultEntitlementResolver: Deterministic, offline, zero-dependency resolver
   representing current released terminal functionality without commercial plans.
"""

from typing import Protocol, Optional, Mapping, Iterable, FrozenSet, Union
from types import MappingProxyType

from api.context.request_context import RequestContext
from api.capabilities.capabilities import CAPABILITIES, LIMITS


class EntitlementSet:
    """
    Immutable set of capabilities and resource limits.

    Guarantees:
    - Boolean capability lookups return strict bool (True or False).
    - Defined limits are non-negative integers.
    - Missing limits return None.
    - Invalid types and negative limits are strictly rejected.
    """

    def __init__(
        self,
        capabilities: Optional[Union[Iterable[str], Mapping[str, bool]]] = None,
        limits: Optional[Mapping[str, int]] = None,
    ) -> None:
        # 1. Parse and validate capabilities
        parsed_caps: set[str] = set()
        if capabilities is not None:
            if isinstance(capabilities, Mapping):
                for cap, enabled in capabilities.items():
                    if not isinstance(cap, str) or not cap.strip():
                        raise ValueError("Capability identifier must be a non-empty string.")
                    if not isinstance(enabled, bool):
                        raise TypeError(f"Capability value for '{cap}' must be a strict boolean, got {type(enabled).__name__}.")
                    if enabled:
                        parsed_caps.add(cap.strip())
            elif isinstance(capabilities, Iterable):
                for cap in capabilities:
                    if not isinstance(cap, str) or not cap.strip():
                        raise ValueError("Capability identifier must be a non-empty string.")
                    parsed_caps.add(cap.strip())
            else:
                raise TypeError("capabilities must be an iterable of strings or a mapping of str to bool.")

        self._capabilities: FrozenSet[str] = frozenset(parsed_caps)

        # 2. Parse and validate limits
        parsed_limits: dict[str, int] = {}
        if limits is not None:
            if not isinstance(limits, Mapping):
                raise TypeError("limits must be a mapping of limit_key (str) to non-negative integer.")
            for key, val in limits.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Limit key must be a non-empty string.")
                # Note: in Python, bool is a subclass of int (isinstance(True, int) is True), so check explicitly
                if isinstance(val, bool) or not isinstance(val, int):
                    raise TypeError(f"Limit value for '{key}' must be an integer, got {type(val).__name__}.")
                if val < 0:
                    raise ValueError(f"Limit value for '{key}' must be non-negative, got {val}.")
                parsed_limits[key.strip()] = val

        self._limits: Mapping[str, int] = MappingProxyType(parsed_limits)

    def can(self, capability: str) -> bool:
        """
        Check whether a capability is enabled.
        Always returns strict boolean True or False.
        """
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError("capability must be a non-empty string.")
        return capability.strip() in self._capabilities

    def get_limit(self, limit_key: str) -> Optional[int]:
        """
        Retrieve a defined integer limit.
        Returns the integer limit if defined, or None if absent.
        """
        if not isinstance(limit_key, str) or not limit_key.strip():
            raise ValueError("limit_key must be a non-empty string.")
        return self._limits.get(limit_key.strip(), None)

    @property
    def capabilities(self) -> FrozenSet[str]:
        """Read-only view of enabled capabilities."""
        return self._capabilities

    @property
    def limits(self) -> Mapping[str, int]:
        """Read-only view of defined limits."""
        return self._limits

    def __repr__(self) -> str:
        return f"EntitlementSet(capabilities={len(self._capabilities)}, limits={len(self._limits)})"


class EntitlementResolver(Protocol):
    """Protocol for entitlement resolution."""

    def resolve(self, context: RequestContext) -> EntitlementSet:
        """Resolve effective EntitlementSet for the provided RequestContext."""
        ...


class DefaultEntitlementResolver:
    """
    Deterministic default resolver for current unauthenticated runtime.

    Grants current product capabilities and baseline limits deterministically
    without consulting external services, billing, environment variables,
    databases, or networks.
    """

    # Baseline released capabilities matching current terminal features
    _DEFAULT_CAPABILITIES: FrozenSet[str] = frozenset({
        "analysis.read",
        "analysis.quant",
        "analysis.simulation",
        "radar.read",
        "radar.advanced_filters",
        "portfolio.read",
        "portfolio.manage",
        "portfolio.risk",
        "journal.read",
        "journal.write",
        "alerts.create",
        "export.csv",
    })

    _DEFAULT_LIMITS: Mapping[str, int] = MappingProxyType({
        "portfolio.max_holdings": 50,
        "portfolio.max_workspaces": 1,
        "alerts.max_active": 10,
        "team.max_members": 1,
        "api.requests_per_day": 500,
    })

    def resolve(self, context: RequestContext) -> EntitlementSet:
        """
        Deterministically resolve entitlements for the context.
        Guarantees: does not mutate context, zero I/O, pure execution.
        """
        if not isinstance(context, RequestContext):
            raise TypeError(f"context must be an instance of RequestContext, got {type(context).__name__}.")

        return EntitlementSet(
            capabilities=self._DEFAULT_CAPABILITIES,
            limits=self._DEFAULT_LIMITS,
        )
