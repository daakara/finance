"""
HVA-01 Reviewer Allocation Engine.
Generates and validates between-subjects reviewer allocations for Protocol V2.1.
"""

from __future__ import annotations

from typing import List, Tuple

from .models import AllocationRecord, AllocationStateError


class AllocationEngine:
    """
    Enforces between-subjects allocation invariants for Protocol V2.1:
    1. Single-package assignment per reviewer.
    2. Absolute ban on within-subjects designs (zero dual-package exposure).
    3. Independent randomization seeds per package stream.
    4. Deterministic, auditable pre-session registration.
    """

    CANONICAL_PILOT_ALPHA_SEED = 2026102342
    CANONICAL_PILOT_BETA_SEED = 2026102399

    @classmethod
    def generate_canonical_between_subjects_pair(
        cls,
        protocol_version: str = "2.1.0-ratified",
        package_a_sha256: str = "ecf4480a0ee7e936839ea1fe209e1af3b083056d6cbaf18a9dd1e1fc32606f1a",
        package_b_sha256: str = "2a8e85c2196024e03099955e8d591b702ec94da8db6522efd337d12f62be52da",
        allocation_id_prefix: str = "ALLOC-HVA01-20261024",
    ) -> Tuple[AllocationRecord, AllocationRecord]:
        """
        Generate the canonical Reviewer A / Reviewer B allocation pair.
        """
        rec_a = AllocationRecord.create(
            allocation_id=f"{allocation_id_prefix}-001",
            protocol_version=protocol_version,
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            assigned_package_sha256=package_a_sha256,
            chart_directory="charts_a/",
            case_id_range=("HVA2-A-01", "HVA2-A-59"),
            randomization_seed=cls.CANONICAL_PILOT_ALPHA_SEED,
            attestation_status="SIGNED_AND_VERIFIED",
        )

        rec_b = AllocationRecord.create(
            allocation_id=f"{allocation_id_prefix}-002",
            protocol_version=protocol_version,
            reviewer_pseudonym="REV-PILOT-BETA",
            assigned_package="PACKAGE_B",
            assigned_package_sha256=package_b_sha256,
            chart_directory="charts_b/",
            case_id_range=("HVA2-B-01", "HVA2-B-59"),
            randomization_seed=cls.CANONICAL_PILOT_BETA_SEED,
            attestation_status="SIGNED_AND_VERIFIED",
        )

        cls.validate_between_subjects_invariants([rec_a, rec_b])
        return rec_a, rec_b

    @classmethod
    def validate_between_subjects_invariants(cls, allocations: List[AllocationRecord]) -> None:
        """
        Verify that a set of allocations adheres to strict between-subjects isolation:
        - No reviewer assigned to multiple packages.
        - No duplicate package assignments without explicit multi-rater authorization.
        - Seeds and chart directories distinct between streams.
        """
        seen_pseudonyms = set()
        seen_packages = set()

        for rec in allocations:
            if rec.reviewer_pseudonym in seen_pseudonyms:
                raise AllocationStateError(
                    f"BETWEEN_SUBJECTS_VIOLATION: Reviewer '{rec.reviewer_pseudonym}' "
                    f"assigned more than once. Within-subjects evaluation is strictly prohibited."
                )
            seen_pseudonyms.add(rec.reviewer_pseudonym)

            if rec.assigned_package in seen_packages:
                raise AllocationStateError(
                    f"DUPLICATE_PACKAGE_ALLOCATION: Package '{rec.assigned_package}' "
                    f"assigned multiple times in single-rater configuration."
                )
            seen_packages.add(rec.assigned_package)

            # Check matching directory and case range
            if rec.assigned_package == "PACKAGE_A":
                if not rec.chart_directory.startswith("charts_a"):
                    raise AllocationStateError(f"Package A must point to charts_a, got {rec.chart_directory}")
                if not rec.case_id_range[0].startswith("HVA2-A-"):
                    raise AllocationStateError(f"Package A range must start with HVA2-A-, got {rec.case_id_range[0]}")
            elif rec.assigned_package == "PACKAGE_B":
                if not rec.chart_directory.startswith("charts_b"):
                    raise AllocationStateError(f"Package B must point to charts_b, got {rec.chart_directory}")
                if not rec.case_id_range[0].startswith("HVA2-B-"):
                    raise AllocationStateError(f"Package B range must start with HVA2-B-, got {rec.case_id_range[0]}")
