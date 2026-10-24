"""
HVA-01 Dual-TSA Fallback Engine and Wire Client.
Implements deterministic two-provider failover state machine with bounded retries,
exponential backoff, explicit failure classification, and complete audit provenance.

GOVERNANCE CLASSIFICATION: TIER 3 — Cryptographic and Evidence-Integrity Controls.
FAIL-CLOSED: Never fabricates timestamps; returns FAILED_CLOSED on exhaustion or fatal error.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cryptography import x509

from .timestamp_wire import (
    DEFAULT_TSA_POLICY_OID,
    ClockDriftExceededError,
    InvalidSignatureError,
    InvalidTimestampTokenError,
    MalformedAsn1Error,
    PKIStatusError,
    RequestResponseBindingError,
    TSAPolicyMismatchError,
    UntrustedTSACertificateError,
    VerifiedTimestamp,
    create_timestamp_request,
    parse_and_verify_timestamp_response,
)


# ============================================================================
# State Machine & Error Classification
# ============================================================================

class TSAFallbackState(str, Enum):
    """Deterministic finite states for dual-TSA provider failover."""
    NOT_STARTED = "NOT_STARTED"
    PRIMARY_REQUESTED = "PRIMARY_REQUESTED"
    PRIMARY_ACCEPTED = "PRIMARY_ACCEPTED"
    PRIMARY_REJECTED = "PRIMARY_REJECTED"
    SECONDARY_REQUESTED = "SECONDARY_REQUESTED"
    SECONDARY_ACCEPTED = "SECONDARY_ACCEPTED"
    SECONDARY_REJECTED = "SECONDARY_REJECTED"
    FAILED_CLOSED = "FAILED_CLOSED"


class TSAFailureClassification(str, Enum):
    """Categorization of failure types to determine fallback eligibility."""
    # Eligible for secondary fallback:
    TRANSPORT_TIMEOUT = "TRANSPORT_TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    PROVIDER_REJECTION = "PROVIDER_REJECTION"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    PROVIDER_UNTRUSTED_OR_EXPIRED = "PROVIDER_UNTRUSTED_OR_EXPIRED"
    PROVIDER_SIGNATURE_INVALID = "PROVIDER_SIGNATURE_INVALID"
    POLICY_MISMATCH = "POLICY_MISMATCH"

    # Ineligible for fallback (immediate FAILED_CLOSED):
    CLIENT_REQUEST_INVALID = "CLIENT_REQUEST_INVALID"
    SYSTEM_CONFIGURATION_ERROR = "SYSTEM_CONFIGURATION_ERROR"


ELIGIBLE_FOR_SECONDARY_FALLBACK = {
    TSAFailureClassification.TRANSPORT_TIMEOUT,
    TSAFailureClassification.TRANSPORT_ERROR,
    TSAFailureClassification.PROVIDER_REJECTION,
    TSAFailureClassification.MALFORMED_RESPONSE,
    TSAFailureClassification.PROVIDER_UNTRUSTED_OR_EXPIRED,
    TSAFailureClassification.PROVIDER_SIGNATURE_INVALID,
    TSAFailureClassification.POLICY_MISMATCH,
}


# ============================================================================
# Exceptions
# ============================================================================

class TSAClientError(Exception):
    """Base exception for TSA client and fallback engine."""
    pass


class TSATransportError(TSAClientError):
    """Raised when transport connection or HTTP communication fails."""
    pass


class TSATimeoutError(TSATransportError):
    """Raised when request times out."""
    pass


class TSAFailoverExhaustedError(TSAClientError):
    """Raised when both primary and secondary providers fail."""
    pass


class TSAFailClosedError(TSAClientError):
    """Raised when fatal, non-recoverable error occurs (immediate fail-closed)."""
    pass


# ============================================================================
# Provider Configuration & Provenance Models
# ============================================================================

@dataclass
class TSAProviderConfig:
    """
    Configuration for an RFC 3161 Timestamp Authority provider endpoint.
    """
    provider_id: str
    endpoint_url: str
    trusted_certificates: List[x509.Certificate]
    policy_oid: str = DEFAULT_TSA_POLICY_OID
    timeout_seconds: float = 2.0
    max_retries: int = 1
    backoff_base_seconds: float = 0.05
    transport_handler: Optional[Callable[[bytes, "TSAProviderConfig"], bytes]] = None

    def execute_transport(self, req_der: bytes) -> bytes:
        """
        Execute transport call to provider endpoint.
        If custom transport_handler is configured, dispatches to it (for simulation/testing).
        """
        if self.transport_handler is not None:
            return self.transport_handler(req_der, self)

        # Default fallback: fail-closed safety guard preventing accidental external socket calls
        raise TSATransportError(
            f"No local synthetic transport_handler provided for {self.provider_id}. "
            f"External live network requests are strictly prohibited under Wave 4C-1."
        )


@dataclass(frozen=True)
class ProviderAttemptRecord:
    """Audit record of a single provider request attempt."""
    provider_id: str
    attempt_number: int
    started_at_utc: str
    duration_ms: float
    outcome: str  # "SUCCESS" or "FAILURE"
    failure_classification: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ProvenanceRecord:
    """Comprehensive evidence provenance for an issued and verified timestamp."""
    request_id: str
    imprint_digest: str
    nonce: Optional[int]
    selected_provider_id: str
    selected_provider_role: str  # "PRIMARY" or "SECONDARY"
    final_state: str
    total_attempts: int
    attempts: List[ProviderAttemptRecord]
    verified_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "imprint_digest": self.imprint_digest,
            "nonce": self.nonce,
            "selected_provider_id": self.selected_provider_id,
            "selected_provider_role": self.selected_provider_role,
            "final_state": self.final_state,
            "total_attempts": self.total_attempts,
            "attempts": [asdict(a) for a in self.attempts],
            "verified_at_utc": self.verified_at_utc,
        }


# ============================================================================
# Dual-TSA Fallback Engine
# ============================================================================

class DualTSAFallbackEngine:
    """
    Deterministic dual-TSA fallback engine.
    Orchestrates Primary -> Secondary failover according to strict eligibility rules.
    """

    def __init__(
        self,
        primary_config: TSAProviderConfig,
        secondary_config: TSAProviderConfig,
    ) -> None:
        self.primary_config = primary_config
        self.secondary_config = secondary_config
        self._state: TSAFallbackState = TSAFallbackState.NOT_STARTED
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def state(self) -> TSAFallbackState:
        return self._state

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def _record_transition(self, to_state: TSAFallbackState, details: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "from_state": self._state.value,
            "to_state": to_state.value,
            "details": details or {},
        }
        self._state = to_state
        self._audit_log.append(entry)

    def request_timestamp(
        self,
        imprint_digest: Union[bytes, str],
        nonce: Optional[int] = None,
        reference_time_utc: Optional[str] = None,
        max_drift_seconds: float = 30.0,
    ) -> Tuple[VerifiedTimestamp, ProvenanceRecord]:
        """
        Request, verify, and bind an RFC 3161 timestamp with deterministic failover.

        Returns:
            Tuple of (VerifiedTimestamp, ProvenanceRecord).

        Raises:
            TSAFailClosedError: on client-level invalid input or unrecoverable error.
            TSAFailoverExhaustedError: when both primary and secondary fail.
        """
        # Validate input imprint
        try:
            req_der = create_timestamp_request(
                imprint_digest=imprint_digest,
                nonce=nonce,
                cert_req=True,
            )
        except Exception as e:
            self._record_transition(TSAFallbackState.FAILED_CLOSED, {"error": str(e), "phase": "request_creation"})
            raise TSAFailClosedError(f"Failed to create valid RFC 3161 request: {e}") from e

        # Extract normalized imprint string and nonce
        if isinstance(imprint_digest, str):
            clean_hex = imprint_digest.strip()
            if not clean_hex.startswith("sha256:"):
                clean_hex = f"sha256:{clean_hex}"
            imprint_str = clean_hex
        else:
            imprint_str = f"sha256:{imprint_digest.hex()}"

        req_id = f"req-{hashlib.sha256(req_der).hexdigest()[:16]}"
        attempts: List[ProviderAttemptRecord] = []

        # ====================================================================
        # Phase 1: Primary Provider
        # ====================================================================
        self._record_transition(TSAFallbackState.PRIMARY_REQUESTED, {"provider": self.primary_config.provider_id})

        primary_success = False
        primary_verified: Optional[VerifiedTimestamp] = None
        primary_last_error: Optional[Exception] = None
        primary_classification: Optional[TSAFailureClassification] = None

        for attempt in range(1, self.primary_config.max_retries + 2):
            t_start = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                resp_der = self.primary_config.execute_transport(req_der)
                verified = parse_and_verify_timestamp_response(
                    resp_der=resp_der,
                    expected_imprint_digest=imprint_str,
                    expected_nonce=nonce,
                    trusted_certs=self.primary_config.trusted_certificates,
                    expected_policy_oid=self.primary_config.policy_oid,
                    reference_time_utc=reference_time_utc,
                    max_drift_seconds=max_drift_seconds,
                )
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                attempts.append(ProviderAttemptRecord(
                    provider_id=self.primary_config.provider_id,
                    attempt_number=attempt,
                    started_at_utc=start_iso,
                    duration_ms=duration_ms,
                    outcome="SUCCESS",
                ))
                primary_verified = verified
                primary_success = True
                break
            except Exception as e:
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                classification = self._classify_error(e)
                attempts.append(ProviderAttemptRecord(
                    provider_id=self.primary_config.provider_id,
                    attempt_number=attempt,
                    started_at_utc=start_iso,
                    duration_ms=duration_ms,
                    outcome="FAILURE",
                    failure_classification=classification.value,
                    error_message=str(e),
                ))
                primary_last_error = e
                primary_classification = classification

                # Bounded backoff retry
                if attempt <= self.primary_config.max_retries and classification in (
                    TSAFailureClassification.TRANSPORT_TIMEOUT,
                    TSAFailureClassification.TRANSPORT_ERROR,
                ):
                    time.sleep(self.primary_config.backoff_base_seconds * (2 ** (attempt - 1)))

        if primary_success and primary_verified is not None:
            self._record_transition(TSAFallbackState.PRIMARY_ACCEPTED, {"provider": self.primary_config.provider_id})
            provenance = ProvenanceRecord(
                request_id=req_id,
                imprint_digest=imprint_str,
                nonce=nonce,
                selected_provider_id=self.primary_config.provider_id,
                selected_provider_role="PRIMARY",
                final_state=self._state.value,
                total_attempts=len(attempts),
                attempts=attempts,
                verified_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            return primary_verified, provenance

        # Primary failed
        self._record_transition(
            TSAFallbackState.PRIMARY_REJECTED,
            {
                "provider": self.primary_config.provider_id,
                "classification": primary_classification.value if primary_classification else "UNKNOWN",
                "error": str(primary_last_error),
            },
        )

        # Check if primary failure is eligible for secondary fallback
        if primary_classification not in ELIGIBLE_FOR_SECONDARY_FALLBACK:
            self._record_transition(TSAFallbackState.FAILED_CLOSED, {
                "reason": "Primary failure not eligible for fallback",
                "classification": primary_classification.value if primary_classification else "UNKNOWN",
            })
            raise TSAFailClosedError(
                f"Primary provider failed with non-failover error: {primary_last_error}"
            ) from primary_last_error

        # ====================================================================
        # Phase 2: Secondary Provider Failover
        # ====================================================================
        self._record_transition(TSAFallbackState.SECONDARY_REQUESTED, {
            "provider": self.secondary_config.provider_id,
            "trigger": primary_classification.value if primary_classification else "UNKNOWN",
        })

        secondary_success = False
        secondary_verified: Optional[VerifiedTimestamp] = None
        secondary_last_error: Optional[Exception] = None
        secondary_classification: Optional[TSAFailureClassification] = None

        for attempt in range(1, self.secondary_config.max_retries + 2):
            t_start = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                resp_der = self.secondary_config.execute_transport(req_der)
                verified = parse_and_verify_timestamp_response(
                    resp_der=resp_der,
                    expected_imprint_digest=imprint_str,
                    expected_nonce=nonce,
                    trusted_certs=self.secondary_config.trusted_certificates,
                    expected_policy_oid=self.secondary_config.policy_oid,
                    reference_time_utc=reference_time_utc,
                    max_drift_seconds=max_drift_seconds,
                )
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                attempts.append(ProviderAttemptRecord(
                    provider_id=self.secondary_config.provider_id,
                    attempt_number=attempt,
                    started_at_utc=start_iso,
                    duration_ms=duration_ms,
                    outcome="SUCCESS",
                ))
                secondary_verified = verified
                secondary_success = True
                break
            except Exception as e:
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                classification = self._classify_error(e)
                attempts.append(ProviderAttemptRecord(
                    provider_id=self.secondary_config.provider_id,
                    attempt_number=attempt,
                    started_at_utc=start_iso,
                    duration_ms=duration_ms,
                    outcome="FAILURE",
                    failure_classification=classification.value,
                    error_message=str(e),
                ))
                secondary_last_error = e
                secondary_classification = classification

                # Bounded backoff retry
                if attempt <= self.secondary_config.max_retries and classification in (
                    TSAFailureClassification.TRANSPORT_TIMEOUT,
                    TSAFailureClassification.TRANSPORT_ERROR,
                ):
                    time.sleep(self.secondary_config.backoff_base_seconds * (2 ** (attempt - 1)))

        if secondary_success and secondary_verified is not None:
            self._record_transition(
                TSAFallbackState.SECONDARY_ACCEPTED,
                {"provider": self.secondary_config.provider_id},
            )
            provenance = ProvenanceRecord(
                request_id=req_id,
                imprint_digest=imprint_str,
                nonce=nonce,
                selected_provider_id=self.secondary_config.provider_id,
                selected_provider_role="SECONDARY",
                final_state=self._state.value,
                total_attempts=len(attempts),
                attempts=attempts,
                verified_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            return secondary_verified, provenance

        # Both providers failed
        self._record_transition(
            TSAFallbackState.SECONDARY_REJECTED,
            {
                "provider": self.secondary_config.provider_id,
                "classification": secondary_classification.value if secondary_classification else "UNKNOWN",
                "error": str(secondary_last_error),
            },
        )
        self._record_transition(TSAFallbackState.FAILED_CLOSED, {
            "reason": "Both primary and secondary providers exhausted",
            "primary_error": str(primary_last_error),
            "secondary_error": str(secondary_last_error),
        })

        p_class_str = primary_classification.value if primary_classification else "NONE"
        s_class_str = secondary_classification.value if secondary_classification else "NONE"
        raise TSAFailoverExhaustedError(
            f"Dual-TSA failover exhausted. Primary failed with ({p_class_str}: {primary_last_error}). "
            f"Secondary failed with ({s_class_str}: {secondary_last_error})."
        )

    def _classify_error(self, err: Exception) -> TSAFailureClassification:
        """Classify exceptions into structured failure categories."""
        if isinstance(err, TSATimeoutError):
            return TSAFailureClassification.TRANSPORT_TIMEOUT
        elif isinstance(err, TSATransportError):
            return TSAFailureClassification.TRANSPORT_ERROR
        elif isinstance(err, PKIStatusError):
            return TSAFailureClassification.PROVIDER_REJECTION
        elif isinstance(err, (MalformedAsn1Error, InvalidTimestampTokenError)):
            return TSAFailureClassification.MALFORMED_RESPONSE
        elif isinstance(err, UntrustedTSACertificateError):
            return TSAFailureClassification.PROVIDER_UNTRUSTED_OR_EXPIRED
        elif isinstance(err, InvalidSignatureError):
            return TSAFailureClassification.PROVIDER_SIGNATURE_INVALID
        elif isinstance(err, TSAPolicyMismatchError):
            return TSAFailureClassification.POLICY_MISMATCH
        elif isinstance(err, RequestResponseBindingError):
            return TSAFailureClassification.MALFORMED_RESPONSE
        elif isinstance(err, ClockDriftExceededError):
            return TSAFailureClassification.PROVIDER_REJECTION
        else:
            return TSAFailureClassification.TRANSPORT_ERROR
