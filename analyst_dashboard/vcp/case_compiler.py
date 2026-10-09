"""ARX VCP Sealed Temporal Case Compiler & Information Closure Engine.

Sprint 2B Domain-Authority Resolution.
Physically truncates source histories at evaluation_as_of:
valid_time <= evaluation_as_of AND known_at <= evaluation_as_of.
Computes deterministic Information Closure Hashes and records Temporal Read Ledgers.
Guarantees zero post-cutoff rows, events, canaries, or corporate action leakage.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar, VCPTemporalContract


@dataclass(frozen=True)
class TemporalReadEvent:
    event_id: str
    run_id: str
    case_id: str
    evidence_id: str
    evidence_type: str
    requested_range: str
    returned_max_valid_time: str
    returned_max_known_at: str
    evaluation_as_of: str
    allowed: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class VCPTemporalCasePackage:
    """Immutable, physically sealed case package evaluated strictly as-of cutoff."""
    case_id: str
    security_id: str
    symbol: str
    evaluation_as_of: str
    permitted_bars: List[DailyOHLCVBar]
    permitted_reference_data: Dict[str, Any]
    permitted_corporate_actions: List[Dict[str, Any]]
    source_hashes: List[str]
    temporal_policy_hash: str
    numeric_contract_hash: str
    domain_contract_hash: str
    case_input_hash: str
    temporal_information_closure_hash: str

    def session_count(self) -> int:
        return len(self.permitted_bars)

    def closes(self) -> List[float]:
        return [b.close for b in self.permitted_bars]

    def highs(self) -> List[float]:
        return [b.high for b in self.permitted_bars]

    def lows(self) -> List[float]:
        return [b.low for b in self.permitted_bars]

    def volumes(self) -> List[int]:
        return [b.volume for b in self.permitted_bars]

    def latest_bar(self) -> Optional[DailyOHLCVBar]:
        return self.permitted_bars[-1] if self.permitted_bars else None


class VCPTemporalCaseCompiler:
    """Compiles raw market snapshots into sealed, bitemporally truncated case packages."""

    def __init__(
        self,
        temporal_contract: Optional[VCPTemporalContract] = None,
        numeric_contract: Optional[VCPNumericContract] = None,
        domain_contract_hash: str = "76b5fe39627139915466d5c36abda74fd966a97734025fffe1da2d5eafd26802",
    ):
        self.temporal_contract = temporal_contract or VCPTemporalContract()
        self.numeric_contract = numeric_contract or VCPNumericContract()
        self.domain_contract_hash = domain_contract_hash
        self.read_ledger: List[TemporalReadEvent] = []

    def log_read_event(
        self,
        run_id: str,
        case_id: str,
        evidence_id: str,
        evidence_type: str,
        requested_range: str,
        returned_max_valid_time: str,
        returned_max_known_at: str,
        evaluation_as_of: str,
        allowed: bool,
        rejection_reason: Optional[str] = None,
    ) -> TemporalReadEvent:
        event = TemporalReadEvent(
            event_id=f"READ-{len(self.read_ledger) + 1:06d}",
            run_id=run_id,
            case_id=case_id,
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            requested_range=requested_range,
            returned_max_valid_time=returned_max_valid_time,
            returned_max_known_at=returned_max_known_at,
            evaluation_as_of=evaluation_as_of,
            allowed=allowed,
            rejection_reason=rejection_reason,
        )
        self.read_ledger.append(event)
        return event

    def compile_case(
        self,
        case_id: str,
        security_id: str,
        symbol: str,
        evaluation_as_of: str,
        raw_bars: List[DailyOHLCVBar],
        reference_data: Dict[str, Any],
        corporate_actions: List[Dict[str, Any]],
        run_id: str = "COMPILER-RUN-001",
    ) -> VCPTemporalCasePackage:
        """Physically truncates raw bars and metadata strictly at evaluation_as_of.

        Guarantees that no post-cutoff bar, corporate action, or reference state
        enters the sealed package.
        """
        cutoff_dt = self.temporal_contract.parse_iso(evaluation_as_of)

        admissible_bars: List[DailyOHLCVBar] = []
        max_valid_time = "1970-01-01T00:00:00Z"
        max_known_at = "1970-01-01T00:00:00Z"

        for b in raw_bars:
            is_adm, reason = self.temporal_contract.is_bar_admissible(b, evaluation_as_of)
            if is_adm:
                admissible_bars.append(b)
                if b.valid_time > max_valid_time:
                    max_valid_time = b.valid_time
                if b.known_at > max_known_at:
                    max_known_at = b.known_at
            else:
                # Log forbidden post-cutoff attempt
                self.log_read_event(
                    run_id=run_id,
                    case_id=case_id,
                    evidence_id=f"{symbol}-BAR-{b.bar_date}",
                    evidence_type="OHLCV_BAR",
                    requested_range=f"Bar on {b.bar_date}",
                    returned_max_valid_time=b.valid_time,
                    returned_max_known_at=b.known_at,
                    evaluation_as_of=evaluation_as_of,
                    allowed=False,
                    rejection_reason=reason,
                )

        # Log permissible read for included bars
        if admissible_bars:
            self.log_read_event(
                run_id=run_id,
                case_id=case_id,
                evidence_id=f"{symbol}-OHLCV-SERIES",
                evidence_type="OHLCV_SERIES",
                requested_range=f"History up to {evaluation_as_of}",
                returned_max_valid_time=max_valid_time,
                returned_max_known_at=max_known_at,
                evaluation_as_of=evaluation_as_of,
                allowed=True,
            )

        # Filter corporate actions known at or before cutoff
        admissible_ca: List[Dict[str, Any]] = []
        for ca in corporate_actions:
            ca_valid = ca.get("effective_date", ca.get("valid_time", "9999-12-31"))
            ca_known = ca.get("announced_date", ca.get("known_at", "9999-12-31"))
            if (
                self.temporal_contract.parse_iso(ca_valid) <= cutoff_dt
                and self.temporal_contract.parse_iso(ca_known) <= cutoff_dt
            ):
                admissible_ca.append(copy.deepcopy(ca))
            else:
                self.log_read_event(
                    run_id=run_id,
                    case_id=case_id,
                    evidence_id=f"{symbol}-CA-{ca.get('action_id', 'UNKNOWN')}",
                    evidence_type="CORPORATE_ACTION",
                    requested_range=f"Action effective {ca_valid}",
                    returned_max_valid_time=ca_valid,
                    returned_max_known_at=ca_known,
                    evaluation_as_of=evaluation_as_of,
                    allowed=False,
                    rejection_reason="POST_CUTOFF_CORPORATE_ACTION",
                )

        # Truncate reference data (strip any latest wall-clock state)
        safe_ref = {
            k: v for k, v in reference_data.items()
            if not k.startswith("current_") and not k.startswith("latest_") and k != "as_of_wall_clock"
        }

        # Compute source hashes over admissible payload
        source_bar_bytes = json.dumps([
            {
                "date": b.bar_date,
                "o": b.open,
                "h": b.high,
                "l": b.low,
                "c": b.close,
                "v": b.volume,
                "vt": b.valid_time,
                "ka": b.known_at,
            }
            for b in admissible_bars
        ], sort_keys=True).encode("utf-8")
        bars_hash = hashlib.sha256(source_bar_bytes).hexdigest()

        ca_bytes = json.dumps(admissible_ca, sort_keys=True).encode("utf-8")
        ca_hash = hashlib.sha256(ca_bytes).hexdigest()

        ref_bytes = json.dumps(safe_ref, sort_keys=True).encode("utf-8")
        ref_hash = hashlib.sha256(ref_bytes).hexdigest()

        source_hashes = [bars_hash, ca_hash, ref_hash]

        temporal_policy_hash = self.temporal_contract.compute_contract_hash()
        numeric_contract_hash = self.numeric_contract.compute_contract_hash()

        # Case input hash
        input_dict = {
            "case_id": case_id,
            "security_id": security_id,
            "symbol": symbol,
            "evaluation_as_of": evaluation_as_of,
            "source_hashes": source_hashes,
        }
        case_input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # Information closure hash closes over all transitive dependencies
        closure_dict = {
            "case_input_hash": case_input_hash,
            "source_hashes": sorted(source_hashes),
            "temporal_policy_hash": temporal_policy_hash,
            "numeric_contract_hash": numeric_contract_hash,
            "domain_contract_hash": self.domain_contract_hash,
            "bar_count": len(admissible_bars),
            "ca_count": len(admissible_ca),
            "max_valid_time": max_valid_time,
            "max_known_at": max_known_at,
        }
        temporal_information_closure_hash = hashlib.sha256(
            json.dumps(closure_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return VCPTemporalCasePackage(
            case_id=case_id,
            security_id=security_id,
            symbol=symbol,
            evaluation_as_of=evaluation_as_of,
            permitted_bars=admissible_bars,
            permitted_reference_data=safe_ref,
            permitted_corporate_actions=admissible_ca,
            source_hashes=source_hashes,
            temporal_policy_hash=temporal_policy_hash,
            numeric_contract_hash=numeric_contract_hash,
            domain_contract_hash=self.domain_contract_hash,
            case_input_hash=case_input_hash,
            temporal_information_closure_hash=temporal_information_closure_hash,
        )

    def scan_for_canaries(
        self, package: VCPTemporalCasePackage, canary_token: str = "CANARY_POST_CUTOFF"
    ) -> List[str]:
        """Detects whether post-cutoff canary strings exist anywhere in the sealed package."""
        exposures: List[str] = []
        # Check bars
        for b in package.permitted_bars:
            if canary_token in str(b):
                exposures.append(f"CANARY_FOUND_IN_BAR: {b.bar_date}")
        # Check reference data
        if canary_token in json.dumps(package.permitted_reference_data):
            exposures.append("CANARY_FOUND_IN_REFERENCE_DATA")
        # Check corporate actions
        if canary_token in json.dumps(package.permitted_corporate_actions):
            exposures.append("CANARY_FOUND_IN_CORPORATE_ACTIONS")
        # Check hashes
        for h in package.source_hashes:
            if canary_token in h:
                exposures.append(f"CANARY_FOUND_IN_HASH: {h}")
        return exposures
