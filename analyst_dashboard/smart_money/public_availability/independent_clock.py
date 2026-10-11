"""
Independent Clock Authority & NTP Measurement — Epoch 001.
Measures real network time offsets against independent reference NTP servers,
derives conservative uncertainty bounds, enforces freshness, and rejects
unverified or out-of-bound local clock authority.
"""

import hashlib
import json
import socket
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.public_availability.models import ClockStatus


# NTP epoch offset: seconds between 1900-01-01 and 1970-01-01
NTP_EPOCH_OFFSET = 2208988800

DEFAULT_INDEPENDENT_NTP_SERVERS = [
    "time.google.com",
    "time.cloudflare.com",
    "time.windows.com",
    "pool.ntp.org",
]

# Governed freshness durations (Section 29)
CLOCK_REFERENCE_MAX_AGE_BEFORE_ARMING_SECONDS = 60.0
CLOCK_REFERENCE_MAX_AGE_DURING_CAPTURE_SECONDS = 300.0

# Conservative error allowances (Section 28)
DEFAULT_NETWORK_ASYMMETRY_ALLOWANCE_MS = 10.0
DEFAULT_OTHER_ERROR_ALLOWANCE_MS = 5.0

# Uncertainty thresholds (Section 2, Public Availability Clock Authority Contract)
MAX_SYNCHRONIZED_BOUND_MS = 100.0
MAX_DEGRADED_BOUND_MS = 500.0


@dataclass(frozen=True)
class ClockSample:
    """Individual reference server measurement sample."""
    server: str
    offset_ms: float
    rtt_ms: float
    uncertainty_ms: float
    stratum: int
    captured_at_utc: str
    raw_packet_hex: str


@dataclass(frozen=True)
class ClockMeasurementResult:
    """
    Certified independent clock measurement result with raw cryptographic evidence.
    """
    measurement_id: str
    captured_at: str
    reference_source_class: str
    reference_servers: List[str]
    samples: List[ClockSample]
    observed_offset_ms: float
    measurement_uncertainty_ms: float
    network_asymmetry_allowance_ms: float
    other_error_allowance_ms: float
    total_bound_ms: float
    clock_status: ClockStatus
    within_policy: bool
    raw_evidence_sha256: str
    is_fresh_for_arming: bool
    is_fresh_during_capture: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "clock_status": self.clock_status.value if hasattr(self.clock_status, "value") else str(self.clock_status),
            "is_fresh_during_capture": self.is_fresh_during_capture,
            "is_fresh_for_arming": self.is_fresh_for_arming,
            "measurement_id": self.measurement_id,
            "measurement_uncertainty_ms": self.measurement_uncertainty_ms,
            "network_asymmetry_allowance_ms": self.network_asymmetry_allowance_ms,
            "observed_offset_ms": self.observed_offset_ms,
            "other_error_allowance_ms": self.other_error_allowance_ms,
            "raw_evidence_sha256": self.raw_evidence_sha256,
            "reference_servers": list(self.reference_servers),
            "reference_source_class": self.reference_source_class,
            "samples": [
                {
                    "captured_at_utc": s.captured_at_utc,
                    "offset_ms": s.offset_ms,
                    "raw_packet_hex": s.raw_packet_hex,
                    "rtt_ms": s.rtt_ms,
                    "server": s.server,
                    "stratum": s.stratum,
                    "uncertainty_ms": s.uncertainty_ms,
                }
                for s in self.samples
            ],
            "total_bound_ms": self.total_bound_ms,
            "within_policy": self.within_policy,
        }


class IndependentClockAuthority:
    """
    Independent Clock Authority (Sections 27, 28, 29, 30, 31).
    Proves independent reference offset, derives conservative bounds, and enforces freshness.
    """

    def __init__(
        self,
        servers: Optional[List[str]] = None,
        timeout_seconds: float = 2.0,
        network_asymmetry_ms: float = DEFAULT_NETWORK_ASYMMETRY_ALLOWANCE_MS,
        other_error_ms: float = DEFAULT_OTHER_ERROR_ALLOWANCE_MS,
    ):
        self.servers = list(servers) if servers is not None else list(DEFAULT_INDEPENDENT_NTP_SERVERS)
        self.timeout = timeout_seconds
        self.network_asymmetry_ms = network_asymmetry_ms
        self.other_error_ms = other_error_ms

    def query_server(self, server: str) -> Optional[ClockSample]:
        """Queries single NTP server over UDP socket with millisecond precision."""
        client = None
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(self.timeout)

            # NTP request packet: LI=0, VN=3, Mode=3 (client) -> 0x1B, rest 47 zero bytes
            req_packet = b"\x1b" + 47 * b"\0"
            t0 = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()

            client.sendto(req_packet, (server, 123))
            resp_data, _ = client.recvfrom(1024)
            t3 = time.time()

            if len(resp_data) < 48:
                return None

            unpacked = struct.unpack("!12I", resp_data[:48])
            stratum = (unpacked[0] >> 16) & 0xFF

            # Receive timestamp (T1 / t_rx): words 8 and 9
            sec_rx = unpacked[8]
            frac_rx = unpacked[9]
            t1 = (sec_rx - NTP_EPOCH_OFFSET) + float(frac_rx) / 2**32

            # Transmit timestamp (T2 / t_tx): words 10 and 11
            sec_tx = unpacked[10]
            frac_tx = unpacked[11]
            t2 = (sec_tx - NTP_EPOCH_OFFSET) + float(frac_tx) / 2**32

            # Standard SNTP RFC 4330 offset and delay calculations:
            # offset = ((T1 - T0) + (T2 - T3)) / 2
            # delay = (T3 - T0) - (T2 - T1)
            offset_sec = ((t1 - t0) + (t2 - t3)) / 2.0
            rtt_sec = max((t3 - t0) - (t2 - t1), 0.0)

            offset_ms = offset_sec * 1000.0
            rtt_ms = rtt_sec * 1000.0
            uncertainty_ms = rtt_ms / 2.0

            return ClockSample(
                server=server,
                offset_ms=round(offset_ms, 3),
                rtt_ms=round(rtt_ms, 3),
                uncertainty_ms=round(uncertainty_ms, 3),
                stratum=stratum,
                captured_at_utc=now_iso,
                raw_packet_hex=resp_data.hex(),
            )
        except Exception:
            return None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def measure_clock(
        self,
        manual_samples: Optional[List[ClockSample]] = None,
        now_override_utc: Optional[str] = None,
    ) -> ClockMeasurementResult:
        """
        Executes real measurement against independent NTP servers (or uses manual samples if provided).
        Computes conservative total bound (Section 28).
        """
        captured_at = now_override_utc or datetime.now(timezone.utc).isoformat()
        samples: List[ClockSample] = []

        if manual_samples is not None:
            samples = list(manual_samples)
        else:
            for srv in self.servers:
                sample = self.query_server(srv)
                if sample is not None:
                    samples.append(sample)

        measurement_id = f"clock_meas_{hashlib.sha256(captured_at.encode()).hexdigest()[:16]}"

        if not samples:
            # Complete absence of independent reference
            raw_evidence = {
                "captured_at": captured_at,
                "error": "NO_INDEPENDENT_NTP_RESPONSE",
                "servers": self.servers,
            }
            raw_sha = hashlib.sha256(canonicalize_json(raw_evidence).encode()).hexdigest()
            return ClockMeasurementResult(
                measurement_id=measurement_id,
                captured_at=captured_at,
                reference_source_class="INDEPENDENT_REFERENCE",
                reference_servers=self.servers,
                samples=[],
                observed_offset_ms=0.0,
                measurement_uncertainty_ms=9999.0,
                network_asymmetry_allowance_ms=self.network_asymmetry_ms,
                other_error_allowance_ms=self.other_error_ms,
                total_bound_ms=9999.0,
                clock_status=ClockStatus.UNTRUSTED,
                within_policy=False,
                raw_evidence_sha256=raw_sha,
                is_fresh_for_arming=False,
                is_fresh_during_capture=False,
                error_message="NO_INDEPENDENT_NTP_REFERENCE_REACHABLE",
            )

        # Median observed offset among valid samples
        sorted_offsets = sorted(s.offset_ms for s in samples)
        mid = len(sorted_offsets) // 2
        median_offset = sorted_offsets[mid] if len(sorted_offsets) % 2 != 0 else (sorted_offsets[mid - 1] + sorted_offsets[mid]) / 2.0

        # Maximum uncertainty among samples
        max_uncertainty = max(s.uncertainty_ms for s in samples)

        # Conservative bound derivation theorem (Section 28):
        # CLOCK_TOTAL_BOUND_MS = abs(CLOCK_OBSERVED_OFFSET_MS)
        #   + CLOCK_MEASUREMENT_UNCERTAINTY_MS
        #   + CLOCK_NETWORK_ASYMMETRY_ALLOWANCE_MS
        #   + CLOCK_OTHER_ERROR_ALLOWANCE_MS
        total_bound = (
            abs(median_offset)
            + max_uncertainty
            + self.network_asymmetry_ms
            + self.other_error_ms
        )

        # Policy evaluation (Section 24 & Contract)
        if total_bound <= MAX_SYNCHRONIZED_BOUND_MS:
            status = ClockStatus.SYNCHRONIZED
            within_policy = True
        elif total_bound <= MAX_DEGRADED_BOUND_MS:
            status = ClockStatus.DEGRADED
            within_policy = True
        else:
            status = ClockStatus.UNTRUSTED
            within_policy = False

        # Freshness evaluation
        is_fresh_arming = self.is_fresh(captured_at, CLOCK_REFERENCE_MAX_AGE_BEFORE_ARMING_SECONDS)
        is_fresh_capture = self.is_fresh(captured_at, CLOCK_REFERENCE_MAX_AGE_DURING_CAPTURE_SECONDS)

        # Raw evidence hash (numbers formatted as strings to satisfy no-float canonicalization)
        evidence_dict = {
            "captured_at": captured_at,
            "median_offset_ms": f"{median_offset:.3f}",
            "network_asymmetry_allowance_ms": f"{self.network_asymmetry_ms:.3f}",
            "other_error_allowance_ms": f"{self.other_error_ms:.3f}",
            "samples": [
                {
                    "offset_ms": f"{s.offset_ms:.3f}",
                    "raw_hex": s.raw_packet_hex,
                    "rtt_ms": f"{s.rtt_ms:.3f}",
                    "server": s.server,
                    "stratum": s.stratum,
                    "uncertainty_ms": f"{s.uncertainty_ms:.3f}",
                }
                for s in samples
            ],
            "total_bound_ms": f"{total_bound:.3f}",
            "uncertainty_ms": f"{max_uncertainty:.3f}",
        }
        raw_evidence_sha = hashlib.sha256(canonicalize_json(evidence_dict).encode()).hexdigest()

        return ClockMeasurementResult(
            measurement_id=measurement_id,
            captured_at=captured_at,
            reference_source_class="INDEPENDENT_REFERENCE",
            reference_servers=[s.server for s in samples],
            samples=samples,
            observed_offset_ms=round(median_offset, 3),
            measurement_uncertainty_ms=round(max_uncertainty, 3),
            network_asymmetry_allowance_ms=self.network_asymmetry_ms,
            other_error_allowance_ms=self.other_error_ms,
            total_bound_ms=round(total_bound, 3),
            clock_status=status,
            within_policy=within_policy,
            raw_evidence_sha256=raw_evidence_sha,
            is_fresh_for_arming=is_fresh_arming,
            is_fresh_during_capture=is_fresh_capture,
        )

    @staticmethod
    def is_fresh(captured_at_iso: str, max_age_seconds: float) -> bool:
        """Evaluates whether measurement is within maximum age."""
        try:
            dt = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = (now - dt).total_seconds()
            return 0.0 <= age <= max_age_seconds
        except Exception:
            return False

    @staticmethod
    def derive_run_start_intervals(
        result: ClockMeasurementResult,
        local_genesis_time_iso: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Derives bounded run genesis interval: [t_start - total_bound, t_start + total_bound]
        (Section 31).
        """
        gen_iso = local_genesis_time_iso or datetime.now(timezone.utc).isoformat()
        dt = datetime.fromisoformat(gen_iso.replace("Z", "+00:00"))
        bound_delta = timedelta(milliseconds=result.total_bound_ms)

        start_interval = (dt - bound_delta).isoformat()
        end_interval = (dt + bound_delta).isoformat()

        return start_interval, end_interval, result.measurement_id
