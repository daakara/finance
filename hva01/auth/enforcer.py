"""
HVA-01 Server-Side Authorization Enforcer.
Provides fail-closed policy evaluation at the trusted execution boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .principal import Principal
from .roles import ROLE_PERMISSIONS, Permission, Role


class PackageAccessViolation(PermissionError):
    """Raised when an attempt is made to access assets outside the assigned package boundary (SC-10)."""
    pass


class RestrictedLabelAccessError(PermissionError):
    """Raised when an attempt is made to access restricted or unblinded evaluation labels (SC-11)."""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails (missing principal, bad token, revoked)."""
    pass


class PrincipalRevokedError(AuthenticationError):
    """Raised when a revoked principal attempts access."""
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when an expired synthetic credential is used."""
    pass


class AuthorizationError(PermissionError):
    """Raised when a principal lacks the required role or permission."""
    pass


class CrossSessionAccessError(AuthorizationError):
    """Raised when a reviewer attempts to access another reviewer's session."""
    pass


class CrossPackageAccessError(PackageAccessViolation, AuthorizationError):
    """Raised when a reviewer attempts to access cases or assets outside assigned package."""
    pass


class UnblindingViolationError(RestrictedLabelAccessError, AuthorizationError):
    """Raised when an unauthorized entity attempts to query ground truth, answer keys, or unblinded data."""
    pass


class SecurityEnforcer:
    """
    Authoritative server-side policy enforcer.
    Evaluates every protected action against the canonical role-permission matrix.
    """

    PROHIBITED_UNBLINDING_TERMS = [
        "RESTRICTED",
        "UNBLIND",
        "MAPPING",
        "ANSWER_KEY",
        "GROUND_TRUTH",
        "ID_MAP",
        "ESCROW",
        "TRUE_LABEL",
        "TICKER",
    ]

    def __init__(self, revoked_principals: Optional[Set[str]] = None) -> None:
        self.revoked_principals: Set[str] = set(revoked_principals or [])

    def revoke_principal(self, principal_id: str) -> None:
        """Add principal ID to server-side revocation registry."""
        self.revoked_principals.add(principal_id)

    def is_revoked(self, principal_id: str) -> bool:
        """Check whether principal is revoked."""
        return principal_id in self.revoked_principals

    def check_authorization(
        self,
        principal: Optional[Principal],
        permission: Permission,
        target_session_id: Optional[str] = None,
        target_package: Optional[str] = None,
        target_case_id: Optional[str] = None,
        asset_path: Optional[str] = None,
    ) -> bool:
        """
        Evaluate authorization for a requested action:
        1. Validates principal presence and non-revocation.
        2. Validates credential expiration.
        3. Enforces RBAC permissions with default-deny.
        4. Enforces strict session isolation (no cross-session data leaks).
        5. Enforces package confinement (Reviewer A != Package B).
        6. Enforces path traversal and unblinding label isolation.
        """
        # 1. Principal existence
        if principal is None:
            raise AuthenticationError("AUTH_REQUIRED: Operation requires authenticated synthetic principal.")

        # Revocation check
        if principal.is_revoked or principal.principal_id in self.revoked_principals:
            raise PrincipalRevokedError(
                f"PRINCIPAL_REVOKED: Principal '{principal.principal_id}' has been formally revoked."
            )

        # 2. Expiration check
        if principal.expires_at_utc:
            try:
                exp_dt = datetime.fromisoformat(principal.expires_at_utc.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                if now_dt > exp_dt:
                    raise TokenExpiredError(
                        f"TOKEN_EXPIRED: Credential for '{principal.principal_id}' expired at {principal.expires_at_utc}."
                    )
            except (ValueError, TypeError):
                raise TokenExpiredError(f"TOKEN_EXPIRED: Invalid expiry format '{principal.expires_at_utc}'.")

        # 3. Deny-by-default RBAC check
        allowed_permissions = ROLE_PERMISSIONS.get(principal.role, frozenset())
        if permission not in allowed_permissions:
            raise AuthorizationError(
                f"PERMISSION_DENIED: Role '{principal.role.value}' lacks permission '{permission.value}'."
            )

        # 4. Reviewer-specific session confinement
        if principal.role == Role.REVIEWER:
            if target_session_id is not None and principal.assigned_session_id is not None:
                if target_session_id != principal.assigned_session_id:
                    raise CrossSessionAccessError(
                        f"CROSS_SESSION_ACCESS_BLOCKED: Reviewer '{principal.principal_id}' assigned to session "
                        f"'{principal.assigned_session_id}' cannot access session '{target_session_id}'."
                    )

            # 5. Reviewer-specific package confinement
            if target_package is not None and principal.assigned_package is not None:
                if target_package != principal.assigned_package:
                    raise CrossPackageAccessError(
                        f"Package access violation: Reviewer '{principal.principal_id}' assigned to "
                        f"'{principal.assigned_package}' cannot access package '{target_package}'."
                    )

        # 6. Asset & Case Path Traversal and Unblinding Isolation
        check_str = f"{target_case_id or ''} {asset_path or ''}".strip()
        if check_str:
            # Traversal guard
            if asset_path:
                norm = asset_path.replace("\\", "/")
                if (
                    ".." in norm
                    or norm.startswith("/")
                    or norm.startswith("\\")
                    or Path(asset_path).is_absolute()
                    or bool(Path(asset_path).drive)
                ):
                    raise AuthorizationError(f"PATH_TRAVERSAL_BLOCKED: Illegal path specification '{asset_path}'.")

            # Unblinding term isolation
            upper_target = check_str.upper()
            for term in self.PROHIBITED_UNBLINDING_TERMS:
                if term in upper_target:
                    # Allow Owner and Custodian only if permission explicitly grants UNBLIND_KEYS_ACCESS
                    if permission != Permission.UNBLIND_KEYS_ACCESS:
                        raise UnblindingViolationError(
                            f"Restricted label access violation: Term '{term}' is restricted and protected by clinical blinding."
                        )

            # Reviewer package boundary check on case ID and path
            if principal.role == Role.REVIEWER and principal.assigned_package:
                if principal.assigned_package == "PACKAGE_A":
                    if target_case_id and target_case_id.startswith("HVA2-B-"):
                        raise CrossPackageAccessError(
                            f"Package access violation: Reviewer assigned to PACKAGE_A cannot access '{target_case_id}' or PACKAGE_B assets."
                        )
                    if asset_path and "charts_b" in asset_path.lower():
                        raise CrossPackageAccessError(
                            f"Package access violation: Reviewer assigned to PACKAGE_A cannot access '{asset_path}'."
                        )
                elif principal.assigned_package == "PACKAGE_B":
                    if target_case_id and target_case_id.startswith("HVA2-A-"):
                        raise CrossPackageAccessError(
                            f"Package access violation: Reviewer assigned to PACKAGE_B cannot access '{target_case_id}' or PACKAGE_A assets."
                        )
                    if asset_path and "charts_a" in asset_path.lower():
                        raise CrossPackageAccessError(
                            f"Package access violation: Reviewer assigned to PACKAGE_B cannot access '{asset_path}'."
                        )

        return True
