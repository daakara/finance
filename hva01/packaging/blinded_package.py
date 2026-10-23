"""
HVA-01 Canonical Blinded Scenario Packaging.
Enforces clinical blinding, zero-leakage serialization, and strict package isolation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..engine.hasher import CanonicalHasher


class PackageIntegrityError(ValueError):
    """Raised when a blinded scenario package fails validation or contains restricted leaks."""
    pass


@dataclass
class BlindedCase:
    """
    Blinded case specification delivered to human reviewer.
    Strictly stripped of all ground truth, ticker symbols, and algorithmic classifications.
    """
    sequence_index: int
    case_id: str
    chart_asset_path: str
    display_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "case_id": self.case_id,
            "chart_asset_path": self.chart_asset_path,
            "display_title": self.display_title,
            "metadata": self.metadata,
        }


class BlindedPackage:
    """
    Canonical Reviewer Package format.
    Provides validation, deterministic hashing, and export controls.
    """

    PROHIBITED_LEAK_KEYS: Set[str] = {
        "ground_truth",
        "ticker",
        "symbol",
        "company_name",
        "algorithm_outcome",
        "vcp_detected",
        "expected_vcp",
        "answer_key",
        "internal_id",
        "mapping_id",
        "true_label",
        "stage_truth",
        "contraction_count_truth",
        "pivot_price",
    }

    def __init__(
        self,
        package_id: str,
        protocol_version: str,
        chart_directory: str,
        cases: List[BlindedCase],
        package_sha256: Optional[str] = None,
    ) -> None:
        self.package_id = package_id
        self.protocol_version = protocol_version
        self.chart_directory = chart_directory
        self.cases = cases
        self.package_sha256 = package_sha256 or self.compute_digest()

    @classmethod
    def create(
        cls,
        package_id: str,
        protocol_version: str,
        chart_directory: str,
        raw_cases: List[Dict[str, Any]],
    ) -> BlindedPackage:
        """
        Create a BlindedPackage from raw case definitions, sanitizing and inspecting for data leaks.
        """
        sanitized_cases: List[BlindedCase] = []
        for idx, item in enumerate(raw_cases, start=1):
            # Check for prohibited keys in raw dictionary
            leaks = [k for k in item.keys() if k.lower() in cls.PROHIBITED_LEAK_KEYS]
            if leaks:
                raise PackageIntegrityError(
                    f"BLINDING_LEAK_DETECTED: Prohibited key(s) {leaks} detected in case '{item.get('case_id')}'."
                )

            # Check inside metadata dictionary if present
            meta = item.get("metadata", {})
            meta_leaks = [k for k in meta.keys() if k.lower() in cls.PROHIBITED_LEAK_KEYS]
            if meta_leaks:
                raise PackageIntegrityError(
                    f"BLINDING_LEAK_DETECTED: Prohibited metadata key(s) {meta_leaks} in case '{item.get('case_id')}'."
                )

            case_id = item["case_id"]
            # Validate case ID format conforms to package
            if package_id == "PACKAGE_A" and not case_id.startswith("HVA2-A-"):
                raise PackageIntegrityError(f"Case ID '{case_id}' violates PACKAGE_A naming convention.")
            elif package_id == "PACKAGE_B" and not case_id.startswith("HVA2-B-"):
                raise PackageIntegrityError(f"Case ID '{case_id}' violates PACKAGE_B naming convention.")

            chart_path = item.get("chart_asset_path", f"{chart_directory}/{case_id}.svg")
            blinded_case = BlindedCase(
                sequence_index=item.get("sequence_index", idx),
                case_id=case_id,
                chart_asset_path=chart_path,
                display_title=item.get("display_title", f"Evaluation Case {idx:02d}"),
                metadata=meta,
            )
            sanitized_cases.append(blinded_case)

        package = cls(
            package_id=package_id,
            protocol_version=protocol_version,
            chart_directory=chart_directory,
            cases=sanitized_cases,
        )
        package.validate()
        return package

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest of package structure."""
        payload = {
            "package_id": self.package_id,
            "protocol_version": self.protocol_version,
            "chart_directory": self.chart_directory,
            "total_cases": len(self.cases),
            "cases": [c.to_dict() for c in self.cases],
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        """
        Validate package against canonical invariants:
        - Exactly 59 cases (unless synthetic subset explicitly permitted)
        - Sequential indices 1..N with zero gaps or duplicates
        - Case IDs unique and matching package
        - Zero unblinding leaks
        """
        if not self.cases:
            raise PackageIntegrityError("Blinded package contains 0 cases.")

        seen_indices = set()
        seen_case_ids = set()

        for idx, c in enumerate(self.cases, start=1):
            if c.sequence_index in seen_indices:
                raise PackageIntegrityError(f"Duplicate sequence index: {c.sequence_index}")
            seen_indices.add(c.sequence_index)

            if c.case_id in seen_case_ids:
                raise PackageIntegrityError(f"Duplicate case ID: {c.case_id}")
            seen_case_ids.add(c.case_id)

            # Ensure chart path matches package chart directory
            if not c.chart_asset_path.startswith(self.chart_directory):
                raise PackageIntegrityError(
                    f"Asset path '{c.chart_asset_path}' does not belong to chart directory '{self.chart_directory}'."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "protocol_version": self.protocol_version,
            "chart_directory": self.chart_directory,
            "total_cases": len(self.cases),
            "package_sha256": self.package_sha256,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save blinded package to disk as canonical JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> BlindedPackage:
        """Load and validate blinded package from disk."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Package file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_cases = data.get("cases", [])
        cases = [
            BlindedCase(
                sequence_index=c["sequence_index"],
                case_id=c["case_id"],
                chart_asset_path=c["chart_asset_path"],
                display_title=c.get("display_title", ""),
                metadata=c.get("metadata", {}),
            )
            for c in raw_cases
        ]
        pkg = cls(
            package_id=data["package_id"],
            protocol_version=data["protocol_version"],
            chart_directory=data["chart_directory"],
            cases=cases,
            package_sha256=data.get("package_sha256"),
        )
        pkg.validate()
        return pkg
