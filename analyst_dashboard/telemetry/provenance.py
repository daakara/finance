"""Corporate Events Telemetry Provenance & Denominator Classification Engine.

Implements the 10-class mutually exclusive provenance taxonomy and
the 8 strict mathematical inclusion predicates defined in
ARX_PR53_NATURAL_EVIDENCE_ELIGIBILITY_SPEC.md.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass

from .corporate_event_contract import (
    CorporateEventTelemetryPayload,
    RiskBucket,
    EvaluationOutcome,
)


class ProvenanceClass(str, Enum):
    """The 10 mutually exclusive provenance classes."""

    NATURAL_PRODUCTION_ELIGIBLE = "NATURAL_PRODUCTION_ELIGIBLE"
    SYNTHETIC_DIAGNOSTIC = "SYNTHETIC_DIAGNOSTIC"
    AUTOMATED_SMOKE_TEST = "AUTOMATED_SMOKE_TEST"
    MANUAL_QA = "MANUAL_QA"
    REPLAY = "REPLAY"
    HISTORICAL_IMPORT = "HISTORICAL_IMPORT"
    DUPLICATE = "DUPLICATE"
    MALFORMED = "MALFORMED"
    UNVERIFIABLE_PROVENANCE = "UNVERIFIABLE_PROVENANCE"
    OUTSIDE_OBSERVATION_EPOCH = "OUTSIDE_OBSERVATION_EPOCH"


@dataclass(frozen=True)
class ObservationEpochBounds:
    """Authoritative release parameters and observation epoch boundaries."""

    RELEASE_SHA: str = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f"
    T0_UTC_STR: str = "2026-10-23T21:45:15Z"
    T72_UTC_STR: str = "2026-10-26T21:45:15Z"
    AUTHORIZED_HOSTS: frozenset[str] = frozenset(
        {
            "arxterminal.com",
            "finance-xp8.pages.dev",
            "b801c8c9.finance-xp8.pages.dev",
            "localhost",
            "127.0.0.1",
        }
    )
    MAX_CLOCK_SKEW_SECONDS: float = 30.0
    MIN_BOT_SCORE: int = 30


AUTOMATED_UA_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ARX-Smoke-Verifier",
        r"curl",
        r"python-requests",
        r"urllib",
        r"Playwright",
        r"Puppeteer",
        r"HeadlessChrome",
        r"pytest",
        r"PostmanRuntime",
        r"k6",
        r"Go-http-client",
    ]
]


@dataclass
class ClassificationResult:
    """Detailed audit result of provenance evaluation."""

    provenance_class: ProvenanceClass
    is_natural_denominator_eligible: bool
    predicate_evaluations: Dict[str, bool]
    exclusion_reason: Optional[str] = None


class ProvenanceClassifier:
    """Evaluates candidate telemetry payloads against the 8 mathematical inclusion predicates."""

    def __init__(self, epoch: Optional[ObservationEpochBounds] = None):
        self.epoch = epoch or ObservationEpochBounds()
        self.t0 = datetime.fromisoformat(self.epoch.T0_UTC_STR.replace("Z", "+00:00"))
        self.t72 = datetime.fromisoformat(self.epoch.T72_UTC_STR.replace("Z", "+00:00"))

    def parse_iso_utc(self, dt_str: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def classify(
        self,
        payload: CorporateEventTelemetryPayload,
        headers: Optional[Dict[str, str]] = None,
        is_known_duplicate: bool = False,
        prior_receipt_time: Optional[datetime] = None,
        collector_received_at: Optional[datetime] = None,
    ) -> ClassificationResult:
        """Deterministically assign one of the 10 provenance classes."""
        headers = headers or {}
        # Normalize header keys to lowercase
        norm_headers = {k.lower(): v for k, v in headers.items()}
        user_agent = norm_headers.get("user-agent", "")

        predicates: Dict[str, bool] = {
            "P1_authorized_release_sha": False,
            "P2_epoch_temporal_validity": False,
            "P3_authorized_host": False,
            "P4_schema_conformance": True,  # Already passed payload model validation
            "P5_synthetic_isolation": False,
            "P6_dedup_uniqueness": not is_known_duplicate,
            "P7_deterministic_outcome": False,
            "P8_edge_monotonic_receipt": False,
        }

        # Check P1: Authorized Release Identity
        if payload.release_sha.lower() == self.epoch.RELEASE_SHA.lower():
            predicates["P1_authorized_release_sha"] = True

        # Check P2: Observation Epoch Temporal Validity (T0 <= occurred <= T72)
        occurred_dt = self.parse_iso_utc(payload.occurred_at_utc)
        if occurred_dt is not None:
            if self.t0 <= occurred_dt <= self.t72:
                predicates["P2_epoch_temporal_validity"] = True

        # Check P3: Authorized Deployment Host
        host = (payload.deployment_host or norm_headers.get("host") or "").lower().split(":")[0]
        if host in self.epoch.AUTHORIZED_HOSTS:
            predicates["P3_authorized_host"] = True

        # Check P5: Synthetic Isolation
        is_synthetic = (
            payload.synthetic_marker
            or norm_headers.get("x-arx-synthetic", "").lower() == "true"
        )
        is_ci = (
            payload.ci_marker
            or norm_headers.get("x-arx-ci", "").lower() == "true"
            or any(pattern.search(user_agent) for pattern in AUTOMATED_UA_PATTERNS)
        )
        is_qa = norm_headers.get("x-arx-qa-session", "").lower() == "true"
        bot_score_ok = (
            payload.bot_score is None or payload.bot_score >= self.epoch.MIN_BOT_SCORE
        )

        if not is_synthetic and not is_ci and not is_qa and bot_score_ok:
            predicates["P5_synthetic_isolation"] = True

        # Check P7: Deterministic Calculation Outcome
        if (
            payload.evaluation_outcome in (EvaluationOutcome.COMPLETED, EvaluationOutcome.UNAVAILABLE)
            and payload.applied_penalty in (-25.0, -8.0, 0.0)
        ):
            predicates["P7_deterministic_outcome"] = True

        # Check P8: Edge Monotonic Receipt (within 30 seconds of client occurrence)
        if collector_received_at is None:
            collector_received_at = datetime.now(timezone.utc)

        if occurred_dt is not None:
            skew_seconds = abs((collector_received_at - occurred_dt).total_seconds())
            if skew_seconds <= self.epoch.MAX_CLOCK_SKEW_SECONDS:
                predicates["P8_edge_monotonic_receipt"] = True

        # Evaluate mutual exclusion hierarchy:
        # 1. SYNTHETIC_DIAGNOSTIC
        if is_synthetic:
            return ClassificationResult(
                provenance_class=ProvenanceClass.SYNTHETIC_DIAGNOSTIC,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason="Explicit synthetic diagnostic marker or header detected",
            )

        # 2. AUTOMATED_SMOKE_TEST
        if is_ci:
            return ClassificationResult(
                provenance_class=ProvenanceClass.AUTOMATED_SMOKE_TEST,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Automated test agent or CI marker detected (UA: {user_agent})",
            )

        # 3. MANUAL_QA
        if is_qa:
            return ClassificationResult(
                provenance_class=ProvenanceClass.MANUAL_QA,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason="Manual QA session header detected",
            )

        # 4. DUPLICATE / REPLAY
        if is_known_duplicate:
            # If retransmitted within 5 seconds of prior receipt -> REPLAY
            if prior_receipt_time and collector_received_at:
                delta = abs((collector_received_at - prior_receipt_time).total_seconds())
                if delta < 5.0:
                    return ClassificationResult(
                        provenance_class=ProvenanceClass.REPLAY,
                        is_natural_denominator_eligible=False,
                        predicate_evaluations=predicates,
                        exclusion_reason=f"Rapid replay detected ({delta:.2f}s < 5.0s)",
                    )
            return ClassificationResult(
                provenance_class=ProvenanceClass.DUPLICATE,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason="Duplicate deduplication_key already registered in ledger",
            )

        # 5. HISTORICAL_IMPORT
        if occurred_dt is not None and occurred_dt < self.t0:
            return ClassificationResult(
                provenance_class=ProvenanceClass.HISTORICAL_IMPORT,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Event timestamp {payload.occurred_at_utc} precedes T0 {self.epoch.T0_UTC_STR}",
            )

        # 6. OUTSIDE_OBSERVATION_EPOCH
        if occurred_dt is not None and occurred_dt > self.t72:
            return ClassificationResult(
                provenance_class=ProvenanceClass.OUTSIDE_OBSERVATION_EPOCH,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Event timestamp {payload.occurred_at_utc} exceeds T72 {self.epoch.T72_UTC_STR}",
            )

        # 7. UNVERIFIABLE_PROVENANCE
        if not predicates["P1_authorized_release_sha"]:
            return ClassificationResult(
                provenance_class=ProvenanceClass.UNVERIFIABLE_PROVENANCE,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Release SHA {payload.release_sha} does not match authorized {self.epoch.RELEASE_SHA}",
            )

        if not predicates["P3_authorized_host"]:
            return ClassificationResult(
                provenance_class=ProvenanceClass.UNVERIFIABLE_PROVENANCE,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Origin host '{host}' not in authorized production host set",
            )

        if not predicates["P8_edge_monotonic_receipt"]:
            return ClassificationResult(
                provenance_class=ProvenanceClass.UNVERIFIABLE_PROVENANCE,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Clock skew between occurred_at and collector receipt exceeds {self.epoch.MAX_CLOCK_SKEW_SECONDS}s",
            )

        if payload.bot_score is not None and payload.bot_score < self.epoch.MIN_BOT_SCORE:
            return ClassificationResult(
                provenance_class=ProvenanceClass.SYNTHETIC_DIAGNOSTIC,
                is_natural_denominator_eligible=False,
                predicate_evaluations=predicates,
                exclusion_reason=f"Bot score {payload.bot_score} below minimum threshold {self.epoch.MIN_BOT_SCORE}",
            )

        # If all 8 predicates evaluate to TRUE
        all_passed = all(predicates.values())
        if all_passed:
            return ClassificationResult(
                provenance_class=ProvenanceClass.NATURAL_PRODUCTION_ELIGIBLE,
                is_natural_denominator_eligible=True,
                predicate_evaluations=predicates,
                exclusion_reason=None,
            )

        return ClassificationResult(
            provenance_class=ProvenanceClass.UNVERIFIABLE_PROVENANCE,
            is_natural_denominator_eligible=False,
            predicate_evaluations=predicates,
            exclusion_reason="Failed general inclusion predicates",
        )
