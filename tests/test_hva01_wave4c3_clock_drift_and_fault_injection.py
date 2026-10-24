"""
HVA-01 Wave 4C-3 Adversarial Test Suite.
Exhaustively qualifies:
- Wall-clock drift detection & monotonic time consistency
- Drift threshold enforcement & clock rollback detection
- Time-source unavailability & fail-closed assertions
- Primary TSA timeout & deterministic failover to secondary
- Secondary TSA timeout & dual-TSA exhaustion
- Synthetic network partition injection & recovery
- Stale timestamp rejection & replay protection
- Fail-closed behavior under conflicting evidence
- Preservation of Gate V3-02 three-state historical admissibility (VALID / INVALID / INDETERMINATE)
"""

import datetime
import hashlib
import time
from typing import Any, Dict, List, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from hva01.witness.client import (
    DualTSAFallbackEngine,
    TSAFailClosedError,
    TSAFailoverExhaustedError,
    TSAFallbackState,
    TSAProviderConfig,
    TSATimeoutError,
    TSATransportError,
)
from hva01.witness.clock_daemon import (
    ClockAnomalyError,
    ClockDriftDaemon,
    ClockDriftExceededError as DaemonDriftExceededError,
    ClockHealthStatus,
    ClockRollbackError,
    TimeSourceUnavailableError,
)
from hva01.witness.fault_harness import (
    AdversarialNetworkHarness,
    ConflictingEvidenceError,
    FaultInjectionMode,
    ReplayAttackDetectedError,
    ReplayProtectionLedger,
)
from hva01.witness.rollover import EvidenceAdmissibilityState, KeyRolloverManager
from hva01.witness.timestamp_wire import (
    ClockDriftExceededError as WireDriftExceededError,
    create_synthetic_timestamp_response,
    create_timestamp_request,
    parse_and_verify_timestamp_response,
    parse_timestamp_request,
)


@pytest.fixture(scope="module")
def pki_setup() -> Dict[str, Any]:
    """Generate synthetic Root CA, Primary TSA, and Secondary TSA keys and certificates."""
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Root CA
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic ARX Root CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=10))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 2. Primary TSA (RSA 2048)
    p_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    p_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Primary Synthetic TSA Desk 01"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    p_cert = (
        x509.CertificateBuilder()
        .subject_name(p_subject)
        .issuer_name(ca_subject)
        .public_key(p_key.public_key())
        .serial_number(1001)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 3. Secondary TSA (ECDSA SECP256R1)
    s_key = ec.generate_private_key(ec.SECP256R1())
    s_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Secondary Synthetic TSA Desk 02"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    s_cert = (
        x509.CertificateBuilder()
        .subject_name(s_subject)
        .issuer_name(ca_subject)
        .public_key(s_key.public_key())
        .serial_number(1002)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    return {
        "ca_cert": ca_cert,
        "primary_key": p_key,
        "primary_cert": p_cert,
        "secondary_key": s_key,
        "secondary_cert": s_cert,
        "trusted_certs": [p_cert, s_cert],
    }


class MockSyntheticClock:
    """Controllable synthetic clock fixture for deterministic time progression testing."""
    def __init__(self, start_wall: float = 1729814400.0, start_mono: float = 1000.0):
        self.wall = start_wall
        self.mono = start_mono

    def time_fn(self) -> float:
        return self.wall

    def mono_fn(self) -> float:
        return self.mono

    def advance(self, wall_seconds: float, mono_seconds: Optional[float] = None) -> None:
        self.wall += wall_seconds
        self.mono += wall_seconds if mono_seconds is None else mono_seconds

    def rollback_wall(self, seconds: float) -> None:
        self.wall -= seconds

    def rollback_mono(self, seconds: float) -> None:
        self.mono -= seconds


def build_test_engine(pki: Dict[str, Any]) -> DualTSAFallbackEngine:
    """Builds a standard dual-provider engine with synthetic mock transports."""
    def primary_handler(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
        req_info = parse_timestamp_request(req_der)
        imprint = f"sha256:{req_info['hashed_message_hex']}"
        return create_synthetic_timestamp_response(
            imprint_digest=imprint,
            signing_key=pki["primary_key"],
            signing_cert=pki["primary_cert"],
            nonce=req_info.get("nonce"),
            serial_number=1001,
        )

    def secondary_handler(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
        req_info = parse_timestamp_request(req_der)
        imprint = f"sha256:{req_info['hashed_message_hex']}"
        return create_synthetic_timestamp_response(
            imprint_digest=imprint,
            signing_key=pki["secondary_key"],
            signing_cert=pki["secondary_cert"],
            nonce=req_info.get("nonce"),
            serial_number=2002,
        )

    primary = TSAProviderConfig(
        provider_id="TSA-Primary-Synthetic",
        endpoint_url="https://mock.tsa.primary.local/tsa",
        trusted_certificates=[pki["primary_cert"]],
        timeout_seconds=2.0,
        max_retries=1,
        backoff_base_seconds=0.01,
        transport_handler=primary_handler,
    )
    secondary = TSAProviderConfig(
        provider_id="TSA-Secondary-Synthetic",
        endpoint_url="https://mock.tsa.secondary.local/tsa",
        trusted_certificates=[pki["secondary_cert"]],
        timeout_seconds=2.0,
        max_retries=1,
        backoff_base_seconds=0.01,
        transport_handler=secondary_handler,
    )
    return DualTSAFallbackEngine(primary, secondary)


# ============================================================================
# Category 1: Clock Drift Daemon & Monotonic Time Consistency
# ============================================================================

class TestWave4C3ClockDriftDaemon:

    def test_c01_healthy_clock_progression(self):
        """Verifies that synchronized wall and monotonic progress remains HEALTHY."""
        clock = MockSyntheticClock()
        daemon = ClockDriftDaemon(
            max_drift_seconds=5.0,
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )
        assert daemon.is_healthy
        assert daemon.status == ClockHealthStatus.HEALTHY

        clock.advance(1.0, 1.0)
        sample = daemon.sample()
        assert sample.status == ClockHealthStatus.HEALTHY
        assert sample.wall_delta == pytest.approx(1.0)
        assert sample.monotonic_delta == pytest.approx(1.0)
        assert sample.drift_delta == pytest.approx(0.0)
        assert daemon.assert_admissible()

    def test_c02_wall_clock_drift_detection(self):
        """Verifies that positive or negative wall-clock divergence triggers drift failure."""
        clock = MockSyntheticClock()
        daemon = ClockDriftDaemon(
            max_drift_seconds=3.0,
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )

        clock.advance(5.0, 1.0)
        with pytest.raises(DaemonDriftExceededError) as exc:
            daemon.sample()
        assert "exceeded limit" in str(exc.value)
        assert daemon.status == ClockHealthStatus.DRIFT_DETECTED
        assert not daemon.is_healthy

        with pytest.raises(DaemonDriftExceededError):
            daemon.assert_admissible()

    def test_c03_wall_clock_rollback_detection(self):
        """Verifies that backward wall-clock jump triggers instantaneous ROLLBACK_DETECTED."""
        clock = MockSyntheticClock()
        daemon = ClockDriftDaemon(
            max_drift_seconds=10.0,
            max_rollback_tolerance_seconds=0.0,
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )

        clock.advance(10.0, 10.0)
        daemon.sample()

        clock.wall -= 2.0
        clock.mono += 1.0

        with pytest.raises(ClockRollbackError) as exc:
            daemon.sample()
        assert "Wall-clock rollback detected" in str(exc.value)
        assert daemon.status == ClockHealthStatus.ROLLBACK_DETECTED

        with pytest.raises(ClockRollbackError):
            daemon.assert_admissible()

    def test_c04_monotonic_reversal_anomaly(self):
        """Verifies that non-physical monotonic reversal triggers immediate fail-closed state."""
        clock = MockSyntheticClock()
        daemon = ClockDriftDaemon(
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )

        clock.advance(5.0, 5.0)
        daemon.sample()

        clock.rollback_mono(2.0)
        with pytest.raises(ClockRollbackError) as exc:
            daemon.sample()
        assert "Monotonic clock reversal detected" in str(exc.value)

    def test_c05_time_source_unavailability(self):
        """Verifies that time source throwing errors transitions daemon to UNAVAILABLE."""
        def faulty_time():
            raise OSError("RTC bus disconnected")

        daemon = ClockDriftDaemon()
        daemon._time_provider = faulty_time

        with pytest.raises(TimeSourceUnavailableError) as exc:
            daemon.sample()
        assert "Time source reading failed" in str(exc.value)
        assert daemon.status == ClockHealthStatus.UNAVAILABLE

        with pytest.raises(TimeSourceUnavailableError):
            daemon.assert_admissible()

    def test_c06_candidate_timestamp_drift_assertion(self):
        """Verifies assert_admissible evaluates candidate external timestamp against current clock."""
        clock = MockSyntheticClock(start_wall=1700000000.0)
        daemon = ClockDriftDaemon(
            max_drift_seconds=30.0,
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )

        valid_ts = datetime.datetime.fromtimestamp(1700000010.0, tz=datetime.timezone.utc).isoformat()
        assert daemon.assert_admissible(valid_ts)

        stale_ts = datetime.datetime.fromtimestamp(1700000000.0 - 45.0, tz=datetime.timezone.utc).isoformat()
        with pytest.raises(DaemonDriftExceededError):
            daemon.assert_admissible(stale_ts)

    def test_c07_daemon_reset_and_recalibration(self):
        """Verifies clean reset and recalibration re-anchors healthy baseline."""
        clock = MockSyntheticClock()
        daemon = ClockDriftDaemon(
            max_drift_seconds=2.0,
            time_provider=clock.time_fn,
            monotonic_provider=clock.mono_fn,
        )

        clock.advance(10.0, 1.0)
        with pytest.raises(DaemonDriftExceededError):
            daemon.sample()
        assert daemon.status == ClockHealthStatus.DRIFT_DETECTED

        daemon.reset()
        assert daemon.status == ClockHealthStatus.HEALTHY
        assert len(daemon.get_audit_history()) == 1


# ============================================================================
# Category 2: Adversarial Network Partition & TSA Fault Injection
# ============================================================================

class TestWave4C3NetworkFaultHarness:

    def test_f01_clean_execution_baseline(self, pki_setup):
        """Verifies normal execution through harness without faults."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.CLEAN)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_001").hexdigest()
        verified, provider = harness.execute_guarded_timestamp(engine, imprint, nonce=101)

        assert verified is not None
        assert provider == "TSA-Primary-Synthetic"
        assert harness.stats.requests_intercepted == 1
        assert harness.stats.timeouts_injected == 0

    def test_f02_primary_timeout_failover_to_secondary(self, pki_setup):
        """Verifies that primary timeout causes deterministic failover to secondary."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.TIMEOUT_PRIMARY)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_timeout_primary").hexdigest()
        verified, provider = harness.execute_guarded_timestamp(engine, imprint, nonce=202)

        assert verified is not None
        assert provider == "TSA-Secondary-Synthetic"
        assert harness.stats.timeouts_injected >= 1
        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED

    def test_f03_secondary_timeout_failover_exhaustion(self, pki_setup):
        """Verifies that failure of secondary provider leads to TSAFailoverExhaustedError."""
        engine = build_test_engine(pki_setup)
        
        # Override primary transport handler to always fail
        def primary_fail(req: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATransportError("Primary outage")
        engine.primary_config.transport_handler = primary_fail

        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.TIMEOUT_SECONDARY)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_secondary_fail").hexdigest()
        with pytest.raises(TSAFailoverExhaustedError) as exc:
            harness.execute_guarded_timestamp(engine, imprint, nonce=303)

        assert "Dual-TSA failover exhausted" in str(exc.value)
        assert engine.state == TSAFallbackState.FAILED_CLOSED

    def test_f04_dual_tsa_exhaustion_timeout_all(self, pki_setup):
        """Verifies that TIMEOUT_ALL exhausts both providers and fails closed."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.TIMEOUT_ALL)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_exhaustion").hexdigest()
        with pytest.raises(TSAFailoverExhaustedError):
            harness.execute_guarded_timestamp(engine, imprint, nonce=404)

        assert engine.state == TSAFallbackState.FAILED_CLOSED
        assert harness.stats.timeouts_injected >= 2

    def test_f05_network_partition_injection_and_recovery(self, pki_setup):
        """Verifies partition injection severs transport, and heal_partition restores service."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness()
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint1 = "sha256:" + hashlib.sha256(b"payload_part_1").hexdigest()
        
        # 1. Inject partition
        harness.inject_partition()
        assert harness.is_partitioned

        with pytest.raises(TSAFailoverExhaustedError):
            harness.execute_guarded_timestamp(engine, imprint1, nonce=501)
        assert harness.stats.partitions_enforced >= 1

        # 2. Heal partition
        harness.heal_partition()
        assert not harness.is_partitioned

        # Re-initialize engine for next request
        engine = build_test_engine(pki_setup)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint2 = "sha256:" + hashlib.sha256(b"payload_part_2").hexdigest()
        verified, provider = harness.execute_guarded_timestamp(engine, imprint2, nonce=502)
        assert verified is not None
        assert provider == "TSA-Primary-Synthetic"

    def test_f06_stale_timestamp_rejection(self, pki_setup):
        """Verifies that responses with timestamp outside drift window are rejected."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(
            mode=FaultInjectionMode.STALE_TIMESTAMP,
            stale_offset_seconds=7200.0,
        )
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_stale").hexdigest()
        ref_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with pytest.raises(TSAFailoverExhaustedError) as exc:
            harness.execute_guarded_timestamp(
                engine, imprint, nonce=601, reference_time_utc=ref_time, max_drift_seconds=30.0
            )

        assert harness.stats.stale_injected >= 1
        assert engine.state == TSAFallbackState.FAILED_CLOSED

    def test_f07_replay_attack_protection(self, pki_setup):
        """Verifies that reusing the same request imprint and nonce triggers ReplayAttackDetectedError."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.CLEAN)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_replay_test").hexdigest()
        nonce = 777888

        verified, _ = harness.execute_guarded_timestamp(engine, imprint, nonce=nonce)
        assert verified is not None

        with pytest.raises(ReplayAttackDetectedError) as exc:
            harness.execute_guarded_timestamp(engine, imprint, nonce=nonce)
        assert "REPLAY_DETECTED" in str(exc.value)

    def test_f08_corrupted_response_handling(self, pki_setup):
        """Verifies that malformed or truncated DER response triggers failover and failure classification."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.CORRUPTED_RESPONSE)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_corrupt").hexdigest()
        with pytest.raises(TSAFailoverExhaustedError):
            harness.execute_guarded_timestamp(engine, imprint, nonce=808)

        assert harness.stats.corruptions_injected >= 1
        assert engine.state == TSAFallbackState.FAILED_CLOSED

    def test_f09_conflicting_evidence_detection(self, pki_setup):
        """Verifies that conflicting signatures/timestamps for same imprint trigger ConflictingEvidenceError."""
        engine = build_test_engine(pki_setup)
        harness = AdversarialNetworkHarness(mode=FaultInjectionMode.CLEAN)
        harness.wrap_engine(
            engine,
            primary_key=pki_setup["primary_key"],
            primary_cert=pki_setup["primary_cert"],
            secondary_key=pki_setup["secondary_key"],
            secondary_cert=pki_setup["secondary_cert"],
        )

        imprint = "sha256:" + hashlib.sha256(b"payload_conflict").hexdigest()

        v1, _ = harness.execute_guarded_timestamp(engine, imprint, nonce=901)
        assert v1 is not None

        harness.set_mode(FaultInjectionMode.CONFLICTING_EVIDENCE)
        with pytest.raises(ConflictingEvidenceError) as exc:
            harness.execute_guarded_timestamp(engine, imprint, nonce=902)
        assert "CONFLICTING_EVIDENCE" in str(exc.value)


# ============================================================================
# Category 3: Gate V3-02 Three-State Historical Validity Preservation
# ============================================================================

class TestWave4C3ThreeStateValidityPreservation:

    def test_v01_valid_before_compromise(self):
        """Timestamp before suspected compromise window evaluates to VALID."""
        token_time = "2026-10-01T12:00:00Z"
        compromise_time = "2026-10-10T00:00:00Z"
        revocation_time = "2026-10-15T00:00:00Z"

        state = AdversarialNetworkHarness.evaluate_historical_admissibility_under_fault(
            token_timestamp_utc=token_time,
            effective_revocation_utc=revocation_time,
            compromise_suspected_utc=compromise_time,
        )
        assert state == EvidenceAdmissibilityState.VALID

    def test_v02_indeterminate_quarantine_in_uncertainty_window(self):
        """Timestamp inside compromise uncertainty window evaluates strictly to INDETERMINATE."""
        token_time = "2026-10-12T15:30:00Z"
        compromise_time = "2026-10-10T00:00:00Z"
        revocation_time = "2026-10-15T00:00:00Z"

        state = AdversarialNetworkHarness.evaluate_historical_admissibility_under_fault(
            token_timestamp_utc=token_time,
            effective_revocation_utc=revocation_time,
            compromise_suspected_utc=compromise_time,
        )
        assert state == EvidenceAdmissibilityState.INDETERMINATE

    def test_v03_invalid_at_or_after_revocation(self):
        """Timestamp at or after effective revocation evaluates strictly to INVALID."""
        token_time = "2026-10-15T00:00:00Z"
        compromise_time = "2026-10-10T00:00:00Z"
        revocation_time = "2026-10-15T00:00:00Z"

        state = AdversarialNetworkHarness.evaluate_historical_admissibility_under_fault(
            token_timestamp_utc=token_time,
            effective_revocation_utc=revocation_time,
            compromise_suspected_utc=compromise_time,
        )
        assert state == EvidenceAdmissibilityState.INVALID

        state_later = AdversarialNetworkHarness.evaluate_historical_admissibility_under_fault(
            token_timestamp_utc="2026-10-20T00:00:00Z",
            effective_revocation_utc=revocation_time,
            compromise_suspected_utc=compromise_time,
        )
        assert state_later == EvidenceAdmissibilityState.INVALID
