"""
HVA-01 Wave 4C-3 Network Partition & Adversarial Fault Injection Harness.
Provides synthetic fault injection for DualTSAFallbackEngine, simulating:
- Primary and secondary TSA timeouts
- Complete dual-TSA network partition and recovery
- Stale timestamp generation and clock skew injection
- Replay attack detection and suppression
- Conflicting cryptographic evidence detection
- Fail-closed verification conforming to Gate V3-02 three-state historical admissibility (VALID/INVALID/INDETERMINATE).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from .client import (
    DualTSAFallbackEngine,
    TSAClientError,
    TSAFailClosedError,
    TSAFailureClassification,
    TSAFailoverExhaustedError,
    TSAFallbackState,
    TSAProviderConfig,
    TSATimeoutError,
    TSATransportError,
)
from .rollover import EvidenceAdmissibilityState, KeyRolloverManager
from .timestamp_wire import (
    ClockDriftExceededError,
    TimestampWireError,
    VerifiedTimestamp,
    create_synthetic_timestamp_response,
    create_timestamp_request,
    parse_and_verify_timestamp_response,
    parse_timestamp_request,
)


class FaultInjectionMode(str, Enum):
    """Fault simulation modes for transport and response behavior."""
    CLEAN = "CLEAN"
    TIMEOUT_PRIMARY = "TIMEOUT_PRIMARY"
    TIMEOUT_SECONDARY = "TIMEOUT_SECONDARY"
    TIMEOUT_ALL = "TIMEOUT_ALL"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    CORRUPTED_RESPONSE = "CORRUPTED_RESPONSE"
    STALE_TIMESTAMP = "STALE_TIMESTAMP"
    REPLAY_PAYLOAD = "REPLAY_PAYLOAD"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class ReplayAttackDetectedError(TSAClientError):
    """Raised when an identical request nonce or digest is re-submitted maliciously."""
    pass


class ConflictingEvidenceError(TSAFailClosedError):
    """Raised when multiple providers or successive attestations produce contradictory cryptographic claims."""
    pass


@dataclass
class FaultHarnessStats:
    """Telemetry counters for adversarial test runs."""
    requests_intercepted: int = 0
    timeouts_injected: int = 0
    partitions_enforced: int = 0
    corruptions_injected: int = 0
    stale_injected: int = 0
    replays_blocked: int = 0
    conflicts_detected: int = 0


class ReplayProtectionLedger:
    """
    Maintains a deterministic cache of processed imprints, nonces, and request signatures
    to detect and block replay attacks.
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._seen_requests: Set[str] = set()
        self._seen_nonces: Set[int] = set()

    def record_and_assert_unique(self, imprint_hex: str, nonce: Optional[int] = None) -> None:
        """
        Record a request and assert it has not been submitted previously.

        Raises:
            ReplayAttackDetectedError: If imprint or nonce has already been consumed.
        """
        req_hash = hashlib.sha256(f"{imprint_hex}:{nonce}".encode("utf-8")).hexdigest()

        if req_hash in self._seen_requests:
            raise ReplayAttackDetectedError(
                f"REPLAY_DETECTED: Request with imprint '{imprint_hex}' and nonce '{nonce}' was previously processed"
            )

        if nonce is not None:
            if nonce in self._seen_nonces:
                raise ReplayAttackDetectedError(
                    f"REPLAY_DETECTED: Nonce '{nonce}' has already been consumed in a prior request"
                )
            self._seen_nonces.add(nonce)

        self._seen_requests.add(req_hash)

        # Evict oldest if exceeding cap
        if len(self._seen_requests) > self.max_entries:
            self._seen_requests.pop()


class AdversarialNetworkHarness:
    """
    Interception harness wrapping DualTSAFallbackEngine providers to inject network partitions,
    timeouts, stale clock responses, replay attacks, and conflicting evidence.
    """

    def __init__(
        self,
        mode: FaultInjectionMode = FaultInjectionMode.CLEAN,
        stale_offset_seconds: float = 3600.0,
    ):
        self.mode = mode
        self.stale_offset_seconds = stale_offset_seconds
        self.is_partitioned = False
        self.stats = FaultHarnessStats()
        self.replay_ledger = ReplayProtectionLedger()

        # Track responses per imprint to detect contradictory evidence
        self._imprint_response_fingerprints: Dict[str, str] = {}

    def inject_partition(self) -> None:
        """Sever all network routes to both primary and secondary TSAs."""
        self.is_partitioned = True
        self.mode = FaultInjectionMode.NETWORK_PARTITION

    def heal_partition(self) -> None:
        """Restore network connectivity and clear partition state."""
        self.is_partitioned = False
        self.mode = FaultInjectionMode.CLEAN

    def set_mode(self, mode: FaultInjectionMode) -> None:
        """Dynamically transition fault injection mode."""
        self.mode = mode
        if mode == FaultInjectionMode.NETWORK_PARTITION:
            self.is_partitioned = True
        else:
            self.is_partitioned = False

    def create_fault_transport(
        self,
        provider_id: str,
        base_transport: Optional[Callable[[bytes, TSAProviderConfig], bytes]] = None,
        signing_key: Optional[Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey]] = None,
        signing_cert: Optional[x509.Certificate] = None,
    ) -> Callable[[bytes, TSAProviderConfig], bytes]:
        """
        Wrap an existing transport callback with active fault injection logic conforming to TSAProviderConfig.
        """
        def _transport(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
            self.stats.requests_intercepted += 1

            # 1. Total network partition check
            if self.is_partitioned or self.mode == FaultInjectionMode.NETWORK_PARTITION:
                self.stats.partitions_enforced += 1
                raise TSATransportError(
                    f"NETWORK_PARTITION: Route to provider '{provider_id}' is partitioned (no route to host)"
                )

            # 2. Timeout injection
            if self.mode == FaultInjectionMode.TIMEOUT_ALL:
                self.stats.timeouts_injected += 1
                raise TSATimeoutError(f"TIMEOUT_ALL: Connection to '{provider_id}' timed out after {cfg.timeout_seconds*1000}ms")

            if self.mode == FaultInjectionMode.TIMEOUT_PRIMARY and "primary" in provider_id.lower():
                self.stats.timeouts_injected += 1
                raise TSATimeoutError(f"TIMEOUT_PRIMARY: Primary provider '{provider_id}' read timed out")

            if self.mode == FaultInjectionMode.TIMEOUT_SECONDARY and "secondary" in provider_id.lower():
                self.stats.timeouts_injected += 1
                raise TSATimeoutError(f"TIMEOUT_SECONDARY: Secondary provider '{provider_id}' connection timed out")

            # 3. Corrupted response injection
            if self.mode == FaultInjectionMode.CORRUPTED_RESPONSE:
                self.stats.corruptions_injected += 1
                # Truncated malformed ASN.1 DER sequence
                return b"\x30\x82\x00\x04\x00\x00\x00\x00"

            # 4. Stale timestamp injection
            if self.mode == FaultInjectionMode.STALE_TIMESTAMP:
                self.stats.stale_injected += 1
                req_info = parse_timestamp_request(req_der)
                stale_dt = datetime.now(timezone.utc) - timedelta(seconds=self.stale_offset_seconds)
                # If key and cert available, generate cryptographically signed stale response
                key = signing_key
                cert = signing_cert or (cfg.trusted_certificates[0] if cfg.trusted_certificates else None)
                if key is not None and cert is not None:
                    imprint = f"sha256:{req_info['hashed_message_hex']}"
                    return create_synthetic_timestamp_response(
                        imprint_digest=imprint,
                        signing_key=key,
                        signing_cert=cert,
                        nonce=req_info.get("nonce"),
                        gen_time_utc=stale_dt.isoformat(),
                    )

            # 5. Conflicting evidence injection
            if self.mode == FaultInjectionMode.CONFLICTING_EVIDENCE:
                self.stats.conflicts_detected += 1
                req_info = parse_timestamp_request(req_der)
                key = signing_key
                cert = signing_cert or (cfg.trusted_certificates[0] if cfg.trusted_certificates else None)
                if key is not None and cert is not None:
                    conflict_dt = datetime.now(timezone.utc) + timedelta(seconds=120)
                    imprint = f"sha256:{req_info['hashed_message_hex']}"
                    return create_synthetic_timestamp_response(
                        imprint_digest=imprint,
                        signing_key=key,
                        signing_cert=cert,
                        nonce=req_info.get("nonce"),
                        gen_time_utc=conflict_dt.isoformat(),
                        serial_number=999999999,
                    )

            # 6. Call base transport for normal operation
            if base_transport is not None:
                return base_transport(req_der, cfg)

            raise TSATransportError(f"No base transport configured for provider '{provider_id}'")

        return _transport

    def wrap_engine(
        self,
        engine: DualTSAFallbackEngine,
        primary_key: Optional[Any] = None,
        primary_cert: Optional[Any] = None,
        secondary_key: Optional[Any] = None,
        secondary_cert: Optional[Any] = None,
    ) -> DualTSAFallbackEngine:
        """
        Wrap an instantiated DualTSAFallbackEngine's primary and secondary transport handlers
        with fault interception.
        """
        engine.primary_config.transport_handler = self.create_fault_transport(
            engine.primary_config.provider_id,
            engine.primary_config.transport_handler,
            signing_key=primary_key,
            signing_cert=primary_cert,
        )
        engine.secondary_config.transport_handler = self.create_fault_transport(
            engine.secondary_config.provider_id,
            engine.secondary_config.transport_handler,
            signing_key=secondary_key,
            signing_cert=secondary_cert,
        )
        return engine

    def execute_guarded_timestamp(
        self,
        engine: DualTSAFallbackEngine,
        imprint_digest: Union[bytes, str],
        nonce: Optional[int] = None,
        reference_time_utc: Optional[str] = None,
        max_drift_seconds: float = 30.0,
    ) -> Tuple[VerifiedTimestamp, str]:
        """
        Execute request_timestamp with active replay protection and conflicting evidence detection.

        Returns:
            Tuple of (VerifiedTimestamp, provider_id).

        Raises:
            ReplayAttackDetectedError: If replay attempt is caught.
            ConflictingEvidenceError: If contradictory attestations are discovered.
            TSAFailClosedError / TSAFailoverExhaustedError: On standard engine failover outcomes.
        """
        imprint_str = imprint_digest if isinstance(imprint_digest, str) else imprint_digest.hex()
        if not imprint_str.startswith("sha256:"):
            imprint_str = f"sha256:{imprint_str}"

        # 1. Assert replay protection
        self.replay_ledger.record_and_assert_unique(imprint_str, nonce)

        # 2. Execute through engine
        verified, provenance = engine.request_timestamp(
            imprint_digest=imprint_digest,
            nonce=nonce,
            reference_time_utc=reference_time_utc,
            max_drift_seconds=max_drift_seconds,
        )

        # 3. Check conflicting evidence across successive calls for same imprint
        current_fp = hashlib.sha256(
            f"{verified.serial_number}:{verified.timestamp_utc}:{provenance.selected_provider_id}".encode("utf-8")
        ).hexdigest()

        if imprint_str in self._imprint_response_fingerprints:
            prior_fp = self._imprint_response_fingerprints[imprint_str]
            if prior_fp != current_fp:
                self.stats.conflicts_detected += 1
                raise ConflictingEvidenceError(
                    f"CONFLICTING_EVIDENCE: Successive responses for imprint '{imprint_str}' yielded conflicting signatures"
                )
        else:
            self._imprint_response_fingerprints[imprint_str] = current_fp

        return verified, provenance.selected_provider_id

    @staticmethod
    def evaluate_historical_admissibility_under_fault(
        token_timestamp_utc: str,
        effective_revocation_utc: str,
        compromise_suspected_utc: Optional[str] = None,
    ) -> str:
        """
        Preserve Gate V3-02 Three-State Historical Validity (VALID, INVALID, INDETERMINATE)
        under fault and partition recovery conditions.
        """
        return KeyRolloverManager.evaluate_three_state_admissibility(
            token_timestamp_utc=token_timestamp_utc,
            effective_revocation_utc=effective_revocation_utc,
            compromise_suspected_utc=compromise_suspected_utc,
        )
