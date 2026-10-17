"""
analyst_dashboard/universe/validator.py

Policy-Independent Input Validation Core for ARX Terminal Universe Inputs.
Guarantees:
- Strict evidence structural validation without relying on SourceSecurity's eligibility-friendly defaults.
- Clear separation between raw inputs, validated values, and downstream admission.
- Fail-closed handling for missing keys, prohibited nulls, malformed values, unknown tokens, and conflicting evidence.
- Point-in-time temporal cutoff assertions to prevent lookahead bias.
- Batch-level collision and conflict detection under caller-supplied uniqueness constraints.
- Pure in-memory execution: zero filesystem writes, zero database queries, zero network calls, zero scheduler/scanner activity.

Trust Boundary:
- Validation success (VALIDATION_PASSED) means ONLY that the supplied validation requirements passed.
- Strategy eligibility, publication readiness, and execution authorization remain strictly unevaluated.
- External evidence references are preserved as external claims, not certified as authoritative truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .contracts import SourceSecurity


# ── Validation Outcomes and Issue Codes ─────────────────────────────────────

class ValidationStatus(str, Enum):
    """
    Validation outcome status for an input record or batch.

    VALIDATION_PASSED: All supplied validation requirements were satisfied.
      NOTE: This asserts ONLY structural validity and conformance with supplied
      requirements. Strategy eligibility, publication readiness, and execution
      authorization remain strictly unevaluated.
    VALIDATION_FAILED: One or more validation requirements were violated.
      Downstream admission is strictly prohibited.
    VALIDATION_UNRESOLVED: Evidence is ambiguous, incomplete, or requires missing
      evaluation context. Downstream admission is strictly prohibited.
    """
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_UNRESOLVED = "VALIDATION_UNRESOLVED"


class ValidationIssueCode(str, Enum):
    """
    Validation-local issue codes.
    Distinct from strategy eligibility reason codes (e.g. INELIGIBLE_ASSET_CLASS).
    """
    # Structural / Schema issues
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    NULL_NOT_PERMITTED = "NULL_NOT_PERMITTED"
    INVALID_DATA_TYPE = "INVALID_DATA_TYPE"
    BOOLEAN_STRING_AMBIGUITY = "BOOLEAN_STRING_AMBIGUITY"
    MALFORMED_VALUE = "MALFORMED_VALUE"
    UNRECOGNIZED_VALUE = "UNRECOGNIZED_VALUE"
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"

    # Temporal issues
    MALFORMED_TIMESTAMP = "MALFORMED_TIMESTAMP"
    LOOKAHEAD_EVIDENCE_UNAVAILABLE = "LOOKAHEAD_EVIDENCE_UNAVAILABLE"
    TEMPORAL_ORDER_VIOLATION = "TEMPORAL_ORDER_VIOLATION"
    MISSING_TEMPORAL_CONTEXT = "MISSING_TEMPORAL_CONTEXT"
    TEMPORAL_STALE_EVIDENCE = "TEMPORAL_STALE_EVIDENCE"

    # Batch / Identity issues
    RECORD_COLLISION = "RECORD_COLLISION"
    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    CONFLICTING_MAPPING = "CONFLICTING_MAPPING"

    # Provenance / Evidence issues
    MALFORMED_EVIDENCE_REFERENCE = "MALFORMED_EVIDENCE_REFERENCE"
    MISSING_EVIDENCE_REFERENCE = "MISSING_EVIDENCE_REFERENCE"
    BATCH_ADMISSION_BLOCKED = "BATCH_ADMISSION_BLOCKED"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldValidationIssue:
    """Represents a discrete validation issue on a specific field or record."""
    field_name: str
    issue_code: ValidationIssueCode
    message: str
    raw_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "issue_code": self.issue_code.value,
            "message": self.message,
            "raw_value": str(self.raw_value) if self.raw_value is not None else None,
        }


@dataclass(frozen=True)
class EvidenceReference:
    """
    Provenance reference for source evidence submitted with an input record.
    External evidence is untrusted until verified by authoritative ingestion pipelines.
    """
    source_authority: str
    source_identifier: str
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    source_hash: Optional[str] = None
    is_external_attestation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalEvaluationContext:
    """
    Evaluation context for point-in-time and anti-lookahead temporal checks.
    All timestamps must be ISO 8601 strings.
    """
    evaluation_cutoff: Optional[str] = None
    target_session_date: Optional[str] = None
    max_directory_lag_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationRequirements:
    """
    Explicit, caller-supplied requirements for validating input records.
    Contains NO implicit eligibility policies or defaults.
    """
    required_fields: Set[str] = field(default_factory=lambda: {
        "security_id",
        "symbol",
        "exchange",
        "security_type",
        "listing_status",
        "currency",
        "country",
        "primary_listing",
        "asset_class",
    })
    permitted_null_fields: Set[str] = field(default_factory=lambda: {
        "listing_date",
        "delisting_date",
        "trading_suspension_date",
        "listing_country",
        "incorporation_country",
        "country_of_incorporation",
        "selected_us_assessment_listing",
    })
    allowed_values: Dict[str, Set[Any]] = field(default_factory=dict)
    type_expectations: Dict[str, type] = field(default_factory=lambda: {
        "security_id": str,
        "symbol": str,
        "exchange": str,
        "security_type": str,
        "listing_status": str,
        "currency": str,
        "country": str,
        "listing_country": str,
        "incorporation_country": str,
        "country_of_incorporation": str,
        "primary_listing": bool,
        "selected_us_assessment_listing": bool,
        "asset_class": str,
        "listing_date": str,
        "delisting_date": str,
        "trading_suspension_date": str,
    })
    uniqueness_keys: Tuple[str, ...] = ("security_id",)
    allow_identical_duplicates: bool = False
    require_evidence_reference: bool = False
    demand_established_evidence: bool = True

    @classmethod
    def create_legacy_requirements(cls, **kwargs) -> ValidationRequirements:
        """Returns standard legacy validation requirements."""
        return cls(**kwargs)

    @classmethod
    def create_listing_jurisdiction_profile(
        cls,
        allowed_listing_countries: Optional[Set[str]] = None,
        require_selected_us_assessment: bool = True,
        allowed_exchanges: Optional[Set[str]] = None,
        allowed_asset_classes: Optional[Set[str]] = None,
        require_evidence_reference: bool = False,
        demand_established_evidence: bool = True,
    ) -> ValidationRequirements:
        """
        Explicit listing-jurisdiction validation profile.

        Validates candidate records based on explicit listing venue jurisdiction (listing_country)
        and designated US assessment line (selected_us_assessment_listing) rather than
        conflating statutory incorporation country or global primary listing.
        """
        allowed_vals: Dict[str, Set[Any]] = {}
        if allowed_listing_countries is not None:
            allowed_vals["listing_country"] = set(allowed_listing_countries)
        if require_selected_us_assessment:
            allowed_vals["selected_us_assessment_listing"] = {True}
        if allowed_exchanges is not None:
            allowed_vals["exchange"] = set(allowed_exchanges)
        if allowed_asset_classes is not None:
            allowed_vals["asset_class"] = set(allowed_asset_classes)

        return cls(
            required_fields={
                "security_id",
                "symbol",
                "exchange",
                "security_type",
                "listing_status",
                "currency",
                "listing_country",
                "selected_us_assessment_listing",
                "asset_class",
            },
            permitted_null_fields={
                "listing_date",
                "delisting_date",
                "trading_suspension_date",
                "country",
                "incorporation_country",
                "country_of_incorporation",
                "primary_listing",
            },
            allowed_values=allowed_vals,
            type_expectations={
                "security_id": str,
                "symbol": str,
                "exchange": str,
                "security_type": str,
                "listing_status": str,
                "currency": str,
                "country": str,
                "listing_country": str,
                "incorporation_country": str,
                "country_of_incorporation": str,
                "primary_listing": bool,
                "selected_us_assessment_listing": bool,
                "asset_class": str,
                "listing_date": str,
                "delisting_date": str,
                "trading_suspension_date": str,
            },
            uniqueness_keys=("security_id",),
            allow_identical_duplicates=False,
            require_evidence_reference=require_evidence_reference,
            demand_established_evidence=demand_established_evidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_fields": sorted(list(self.required_fields)),
            "permitted_null_fields": sorted(list(self.permitted_null_fields)),
            "allowed_values": {k: sorted(list(v)) for k, v in self.allowed_values.items()},
            "uniqueness_keys": list(self.uniqueness_keys),
            "allow_identical_duplicates": self.allow_identical_duplicates,
            "require_evidence_reference": self.require_evidence_reference,
            "demand_established_evidence": self.demand_established_evidence,
        }


@dataclass
class RecordValidationResult:
    """
    Structured validation result for a single input record.

    Invariants:
    - If status != VALIDATION_PASSED, downstream_admission_permitted is False.
    - If status != VALIDATION_PASSED, validated_values is None (never exposes unvalidated data).
    - downstream_admission_permitted == True means ONLY that validation requirements passed;
      it does NOT indicate strategy eligibility or execution authorization.
    """
    raw_record: Dict[str, Any]
    status: ValidationStatus
    issues: List[FieldValidationIssue] = field(default_factory=list)
    validated_values: Optional[Dict[str, Any]] = None
    evidence_references: List[EvidenceReference] = field(default_factory=list)
    downstream_admission_permitted: bool = False
    requirements: Optional[ValidationRequirements] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "issues": [i.to_dict() for i in self.issues],
            "validated_values": self.validated_values,
            "evidence_references": [e.to_dict() for e in self.evidence_references],
            "downstream_admission_permitted": self.downstream_admission_permitted,
        }


@dataclass
class BatchValidationResult:
    """Structured validation result for a batch of input records."""
    total_input_count: int
    passed_count: int
    failed_count: int
    unresolved_count: int
    records: List[RecordValidationResult]
    batch_issues: List[FieldValidationIssue] = field(default_factory=list)
    is_batch_valid: bool = False

    def get_validated_records(self) -> List[Dict[str, Any]]:
        """
        Returns validated values for all records if and only if the batch is completely valid.
        Raises ValueError if batch validation failed.
        """
        if not self.is_batch_valid:
            raise ValueError(
                f"Cannot retrieve validated records: batch validation failed "
                f"({self.failed_count} failed, {self.unresolved_count} unresolved, "
                f"{len(self.batch_issues)} batch-level issues)."
            )
        return [r.validated_values for r in self.records if r.validated_values is not None]

    def to_source_securities(self) -> List[SourceSecurity]:
        """
        Converts all records in a completely validated batch to SourceSecurity instances.
        Raises ValueError if batch validation failed.
        """
        if not self.is_batch_valid:
            raise ValueError(
                f"Cannot convert records from a failed batch to SourceSecurity. "
                f"Batch validation failed ({self.failed_count} failed, "
                f"{self.unresolved_count} unresolved, {len(self.batch_issues)} batch-level issues)."
            )
        return [to_source_security(r) for r in self.records]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_count": self.total_input_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "unresolved_count": self.unresolved_count,
            "is_batch_valid": self.is_batch_valid,
            "batch_issues": [i.to_dict() for i in self.batch_issues],
            "records": [r.to_dict() for r in self.records],
        }


# ── Parsing and Temporal Helpers ─────────────────────────────────────────────

_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Parses an ISO 8601 string into a timezone-aware UTC datetime."""
    if not isinstance(ts_str, str) or not ts_str.strip():
        return None
    cleaned = ts_str.strip()
    try:
        # Standardize trailing Z to UTC offset
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_date_string(date_str: str) -> Optional[datetime]:
    """Validates YYYY-MM-DD Gregorian date syntax and parses to datetime."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    cleaned = date_str.strip()
    if not _DATE_REGEX.match(cleaned):
        return None
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError:
        return None


# ── Core Validation Logic ────────────────────────────────────────────────────

class UniverseInputValidator:
    """
    Pure in-memory input validator for incoming universe source evidence.
    Enforces structural completeness, explicit values, and temporal anti-lookahead.
    """

    @classmethod
    def validate_record(
        cls,
        raw_record: Dict[str, Any],
        requirements: ValidationRequirements,
        temporal_context: Optional[TemporalEvaluationContext] = None,
        evidence: Optional[List[EvidenceReference]] = None,
    ) -> RecordValidationResult:
        """
        Validates a single raw input record against explicit requirements.
        Pure in-memory function: no side effects, no external dependencies.
        """
        issues: List[FieldValidationIssue] = []
        validated_values: Dict[str, Any] = {}
        evidence_list = list(evidence or [])

        # 1. Evidence Reference Completeness
        if requirements.require_evidence_reference and not evidence_list:
            issues.append(FieldValidationIssue(
                field_name="__evidence__",
                issue_code=ValidationIssueCode.MISSING_EVIDENCE_REFERENCE,
                message="Input record missing required provenance evidence reference.",
            ))

        for ev in evidence_list:
            if not isinstance(ev, EvidenceReference):
                issues.append(FieldValidationIssue(
                    field_name="__evidence__",
                    issue_code=ValidationIssueCode.MALFORMED_EVIDENCE_REFERENCE,
                    message="Supplied evidence is not an instance of EvidenceReference.",
                    raw_value=ev,
                ))
            elif not ev.source_authority or not ev.source_identifier:
                issues.append(FieldValidationIssue(
                    field_name="__evidence__",
                    issue_code=ValidationIssueCode.MALFORMED_EVIDENCE_REFERENCE,
                    message="EvidenceReference missing source_authority or source_identifier.",
                    raw_value=ev,
                ))

        # Check for unadjudicated source conflicts or ambiguity passed from adapter/resolver
        if "__source_conflict__" in raw_record and raw_record["__source_conflict__"]:
            conf_val = raw_record["__source_conflict__"]
            issues.append(FieldValidationIssue(
                field_name="__source_conflict__",
                issue_code=ValidationIssueCode.CONFLICTING_MAPPING,
                message=f"Unadjudicated source conflict or ambiguity cannot be cleared by overlay: {conf_val}",
                raw_value=conf_val,
            ))

        # 2. Field-by-Field Structural & Value Checks
        # Evaluate all required fields
        for field_name in requirements.required_fields:
            if field_name not in raw_record:
                issues.append(FieldValidationIssue(
                    field_name=field_name,
                    issue_code=ValidationIssueCode.MISSING_REQUIRED_FIELD,
                    message=f"Required field '{field_name}' is absent from raw record.",
                ))

        # Evaluate each supplied attribute in raw_record
        for field_name, raw_val in raw_record.items():
            if field_name.startswith("__"):
                continue
            expected_type = requirements.type_expectations.get(field_name)
            is_permitted_null = field_name in requirements.permitted_null_fields

            # 2A. Null Handling
            if raw_val is None:
                if not is_permitted_null:
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.NULL_NOT_PERMITTED,
                        message=f"Field '{field_name}' is None, but null is not permitted for this field.",
                        raw_value=raw_val,
                    ))
                else:
                    validated_values[field_name] = None
                continue

            # 2B. String-Specific Checks (Empty / Whitespace / UNKNOWN)
            if isinstance(raw_val, str):
                stripped = raw_val.strip()
                if stripped == "":
                    if not is_permitted_null:
                        issues.append(FieldValidationIssue(
                            field_name=field_name,
                            issue_code=ValidationIssueCode.NULL_NOT_PERMITTED,
                            message=f"Field '{field_name}' is an empty or whitespace-only string.",
                            raw_value=raw_val,
                        ))
                    else:
                        validated_values[field_name] = None
                    continue

                if requirements.demand_established_evidence and stripped.upper() in {
                    "UNKNOWN", "N/A", "NULL", "NONE", "UNDEFINED"
                }:
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.UNKNOWN_EVIDENCE,
                        message=f"Field '{field_name}' contains explicit uncertified token '{stripped}'.",
                        raw_value=raw_val,
                    ))
                    continue

            # 2C. Type Checks & Boolean/String Confusion
            if expected_type is bool:
                if isinstance(raw_val, bool):
                    validated_values[field_name] = raw_val
                elif isinstance(raw_val, str):
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.BOOLEAN_STRING_AMBIGUITY,
                        message=(
                            f"Field '{field_name}' expects strict boolean, got string '{raw_val}'. "
                            f"String-to-boolean coercion is prohibited."
                        ),
                        raw_value=raw_val,
                    ))
                else:
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_DATA_TYPE,
                        message=f"Field '{field_name}' expects boolean, got {type(raw_val).__name__}.",
                        raw_value=raw_val,
                    ))
                continue

            elif expected_type is str:
                if not isinstance(raw_val, str):
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_DATA_TYPE,
                        message=f"Field '{field_name}' expects string, got {type(raw_val).__name__}.",
                        raw_value=raw_val,
                    ))
                    continue
                # Deterministic whitespace trimming
                norm_str = raw_val.strip()
                # Uppercase standard ticker/exchange/country/currency fields deterministically
                if field_name in {"symbol", "exchange", "currency", "country", "listing_country", "incorporation_country", "country_of_incorporation", "asset_class", "security_type", "listing_status"}:
                    norm_str = norm_str.upper()
                validated_values[field_name] = norm_str

            elif expected_type is int:
                if not isinstance(raw_val, int) or isinstance(raw_val, bool):
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_DATA_TYPE,
                        message=f"Field '{field_name}' expects int, got {type(raw_val).__name__}.",
                        raw_value=raw_val,
                    ))
                    continue
                validated_values[field_name] = raw_val

            elif expected_type is float:
                if not isinstance(raw_val, (int, float)) or isinstance(raw_val, bool):
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_DATA_TYPE,
                        message=f"Field '{field_name}' expects float, got {type(raw_val).__name__}.",
                        raw_value=raw_val,
                    ))
                    continue
                validated_values[field_name] = float(raw_val)

            else:
                # No specific type expectation registered; preserve as-is
                validated_values[field_name] = raw_val

            # 2D. Allowed Vocabulary Checks
            if field_name in requirements.allowed_values:
                allowed_set = requirements.allowed_values[field_name]
                current_val = validated_values.get(field_name)
                if current_val is not None and current_val not in allowed_set:
                    issues.append(FieldValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.UNRECOGNIZED_VALUE,
                        message=f"Field '{field_name}' value '{current_val}' not in allowed vocabulary.",
                        raw_value=raw_val,
                    ))

            # 2E. Date Format Validation for Date Fields
            if field_name in {"listing_date", "delisting_date", "trading_suspension_date"}:
                val_to_check = validated_values.get(field_name)
                if val_to_check is not None:
                    parsed_d = _parse_date_string(val_to_check)
                    if parsed_d is None:
                        issues.append(FieldValidationIssue(
                            field_name=field_name,
                            issue_code=ValidationIssueCode.MALFORMED_VALUE,
                            message=f"Field '{field_name}' date '{val_to_check}' is not a valid YYYY-MM-DD date.",
                            raw_value=raw_val,
                        ))

        # 3. Inter-Field Temporal Ordering Checks
        listing_date_str = validated_values.get("listing_date")
        delisting_date_str = validated_values.get("delisting_date")
        if listing_date_str and delisting_date_str:
            d_list = _parse_date_string(listing_date_str)
            d_delist = _parse_date_string(delisting_date_str)
            if d_list and d_delist and d_delist <= d_list:
                issues.append(FieldValidationIssue(
                    field_name="delisting_date",
                    issue_code=ValidationIssueCode.TEMPORAL_ORDER_VIOLATION,
                    message=(
                        f"delisting_date ({delisting_date_str}) must be strictly after "
                        f"listing_date ({listing_date_str})."
                    ),
                    raw_value=delisting_date_str,
                ))

        # 4. Temporal Evaluation Context & Anti-Lookahead Checks
        if temporal_context is not None:
            # Check Evidence Availability vs Cutoff
            if temporal_context.evaluation_cutoff:
                cutoff_dt = _parse_iso_timestamp(temporal_context.evaluation_cutoff)
                if cutoff_dt is None:
                    issues.append(FieldValidationIssue(
                        field_name="__temporal_context__",
                        issue_code=ValidationIssueCode.MALFORMED_TIMESTAMP,
                        message=(
                            f"Evaluation cutoff '{temporal_context.evaluation_cutoff}' "
                            f"is not a valid ISO 8601 timestamp."
                        ),
                        raw_value=temporal_context.evaluation_cutoff,
                    ))
                else:
                    for ev in evidence_list:
                        if ev.published_at:
                            pub_dt = _parse_iso_timestamp(ev.published_at)
                            if pub_dt is None:
                                issues.append(FieldValidationIssue(
                                    field_name="published_at",
                                    issue_code=ValidationIssueCode.MALFORMED_TIMESTAMP,
                                    message=f"Evidence published_at '{ev.published_at}' is not a valid ISO 8601 timestamp.",
                                    raw_value=ev.published_at,
                                ))
                            elif pub_dt > cutoff_dt:
                                issues.append(FieldValidationIssue(
                                    field_name="published_at",
                                    issue_code=ValidationIssueCode.LOOKAHEAD_EVIDENCE_UNAVAILABLE,
                                    message=(
                                        f"Evidence publication timestamp '{ev.published_at}' is strictly after "
                                        f"evaluation cutoff '{temporal_context.evaluation_cutoff}'. "
                                        f"Lookahead evidence is rejected fail-closed."
                                    ),
                                    raw_value=ev.published_at,
                                ))

            # Check directory freshness if max_directory_lag_seconds is explicitly configured
            if temporal_context.max_directory_lag_seconds is not None and temporal_context.evaluation_cutoff:
                cutoff_dt = _parse_iso_timestamp(temporal_context.evaluation_cutoff)
                if cutoff_dt:
                    for ev in evidence_list:
                        if ev.published_at:
                            pub_dt = _parse_iso_timestamp(ev.published_at)
                            if pub_dt:
                                lag_seconds = (cutoff_dt - pub_dt).total_seconds()
                                if lag_seconds > temporal_context.max_directory_lag_seconds:
                                    issues.append(FieldValidationIssue(
                                        field_name="published_at",
                                        issue_code=ValidationIssueCode.TEMPORAL_STALE_EVIDENCE,
                                        message=(
                                            f"Evidence age ({lag_seconds:.1f}s) exceeds maximum allowable "
                                            f"directory lag ({temporal_context.max_directory_lag_seconds:.1f}s)."
                                        ),
                                        raw_value=ev.published_at,
                                    ))

        # 5. Outcome Determination
        if issues:
            has_unresolved = any(
                i.issue_code in {
                    ValidationIssueCode.UNKNOWN_EVIDENCE,
                    ValidationIssueCode.MISSING_TEMPORAL_CONTEXT,
                }
                for i in issues
            )
            status = ValidationStatus.VALIDATION_UNRESOLVED if has_unresolved else ValidationStatus.VALIDATION_FAILED
            return RecordValidationResult(
                raw_record=raw_record,
                status=status,
                issues=issues,
                validated_values=None,  # Never expose partial/failed values
                evidence_references=evidence_list,
                downstream_admission_permitted=False,
                requirements=requirements,
            )

        return RecordValidationResult(
            raw_record=raw_record,
            status=ValidationStatus.VALIDATION_PASSED,
            issues=[],
            validated_values=validated_values,
            evidence_references=evidence_list,
            downstream_admission_permitted=True,
            requirements=requirements,
        )

    @classmethod
    def validate_batch(
        cls,
        raw_records: List[Dict[str, Any]],
        requirements: ValidationRequirements,
        temporal_context: Optional[TemporalEvaluationContext] = None,
        evidence_map: Optional[Dict[str, List[EvidenceReference]]] = None,
    ) -> BatchValidationResult:
        """
        Validates a batch of raw records, enforcing per-record requirements
        and batch-level uniqueness/collision constraints.
        """
        record_results: List[RecordValidationResult] = []
        batch_issues: List[FieldValidationIssue] = []

        # 0. Empty input batch guard
        if not raw_records:
            batch_issues.append(FieldValidationIssue(
                field_name="__batch__",
                issue_code=ValidationIssueCode.BATCH_ADMISSION_BLOCKED,
                message="Empty input batch cannot produce a valid or certified universe.",
            ))
            return BatchValidationResult(
                total_input_count=0,
                passed_count=0,
                failed_count=0,
                unresolved_count=0,
                records=[],
                batch_issues=batch_issues,
                is_batch_valid=False,
            )

        # 1. Per-Record Validation
        for idx, raw_record in enumerate(raw_records):
            # Derive evidence lookup key using security_id or symbol
            rec_id = str(raw_record.get("security_id") or raw_record.get("symbol") or idx)
            rec_evidence = (evidence_map or {}).get(rec_id, [])

            res = cls.validate_record(
                raw_record=raw_record,
                requirements=requirements,
                temporal_context=temporal_context,
                evidence=rec_evidence,
            )
            record_results.append(res)

        # 2. Batch-Level Collision & Conflict Detection
        # Group valid records by uniqueness key tuple
        keyed_groups: Dict[Tuple[Any, ...], List[Tuple[int, RecordValidationResult]]] = {}
        for idx, res in enumerate(record_results):
            if res.validated_values is not None:
                # Construct composite uniqueness key tuple from validated values
                key_tuple = tuple(res.validated_values.get(k) for k in requirements.uniqueness_keys)
                keyed_groups.setdefault(key_tuple, []).append((idx, res))

        # Check for collisions and duplicates
        for key_tuple, group in keyed_groups.items():
            if len(group) > 1:
                # Compare full normalized payloads
                first_payload = group[0][1].validated_values
                is_identical = all(r[1].validated_values == first_payload for r in group)

                if is_identical:
                    if not requirements.allow_identical_duplicates:
                        issue = FieldValidationIssue(
                            field_name=",".join(requirements.uniqueness_keys),
                            issue_code=ValidationIssueCode.DUPLICATE_RECORD,
                            message=(
                                f"Duplicate identical records ({len(group)} instances) detected for "
                                f"uniqueness key {key_tuple}."
                            ),
                            raw_value=key_tuple,
                        )
                        batch_issues.append(issue)
                        # Demote records fail-closed
                        for idx, rec in group:
                            rec.status = ValidationStatus.VALIDATION_FAILED
                            rec.issues.append(issue)
                            rec.validated_values = None
                            rec.downstream_admission_permitted = False
                else:
                    # Conflicting records with different attributes under the same identity key
                    differing_fields = set()
                    for r in group[1:]:
                        for k, v in r[1].validated_values.items():
                            if first_payload.get(k) != v:
                                differing_fields.add(k)
                    issue = FieldValidationIssue(
                        field_name=",".join(differing_fields) or ",".join(requirements.uniqueness_keys),
                        issue_code=ValidationIssueCode.RECORD_COLLISION,
                        message=(
                            f"Conflicting records ({len(group)} instances) colliding on uniqueness key "
                            f"{key_tuple}. Differing attributes: {sorted(list(differing_fields))}."
                        ),
                        raw_value=key_tuple,
                    )
                    batch_issues.append(issue)
                    for idx, rec in group:
                        rec.status = ValidationStatus.VALIDATION_FAILED
                        rec.issues.append(issue)
                        rec.validated_values = None
                        rec.downstream_admission_permitted = False

        # 3. Aggregate Batch Counts
        passed_count = sum(1 for r in record_results if r.status == ValidationStatus.VALIDATION_PASSED)
        failed_count = sum(1 for r in record_results if r.status == ValidationStatus.VALIDATION_FAILED)
        unresolved_count = sum(1 for r in record_results if r.status == ValidationStatus.VALIDATION_UNRESOLVED)
        is_batch_valid = (
            len(raw_records) > 0
            and failed_count == 0
            and unresolved_count == 0
            and len(batch_issues) == 0
            and len(raw_records) == passed_count
        )

        if not is_batch_valid:
            # Atomic batch contract: if the batch fails, no constituent record may be admitted.
            for r in record_results:
                if r.status == ValidationStatus.VALIDATION_PASSED:
                    r.downstream_admission_permitted = False
                    r.issues.append(FieldValidationIssue(
                        field_name="__batch__",
                        issue_code=ValidationIssueCode.BATCH_ADMISSION_BLOCKED,
                        message=(
                            "Record is structurally valid, but batch validation failed. "
                            "Admission prohibited under atomic batch contract."
                        ),
                    ))
                    r.validated_values = None

        return BatchValidationResult(
            total_input_count=len(raw_records),
            passed_count=passed_count,
            failed_count=failed_count,
            unresolved_count=unresolved_count,
            records=record_results,
            batch_issues=batch_issues,
            is_batch_valid=is_batch_valid,
        )


# ── Explicit Builder Conversion Boundary Helper ──────────────────────────────

def to_source_security(record_result: RecordValidationResult) -> SourceSecurity:
    """
    Explicit, guarded conversion helper from a validated record to SourceSecurity.

    Guarantees:
    - Permitted ONLY if record_result.status == VALIDATION_PASSED and downstream_admission_permitted == True.
    - Every consumed field must be explicitly present in validated_values.
    - Inheriting SourceSecurity's eligibility-friendly defaults is strictly prohibited.
    - Does NOT execute universe builder, does NOT write to store, does NOT activate scanners.
    """
    if not record_result.downstream_admission_permitted or record_result.status != ValidationStatus.VALIDATION_PASSED:
        raise ValueError(
            f"Cannot convert unvalidated or failed record to SourceSecurity. "
            f"ValidationStatus: {record_result.status}, admission_permitted: {record_result.downstream_admission_permitted}."
        )

    vals = record_result.validated_values
    if vals is None:
        raise ValueError("Record passed validation but validated_values is None.")

    reqs = record_result.requirements
    is_listing_profile = (
        reqs is not None
        and "listing_country" in reqs.required_fields
        and "country" not in reqs.required_fields
    )

    if is_listing_profile:
        mandatory_fields = [
            "security_id",
            "symbol",
            "exchange",
            "security_type",
            "listing_status",
            "currency",
            "listing_country",
            "selected_us_assessment_listing",
            "asset_class",
        ]
    else:
        mandatory_fields = [
            "security_id",
            "symbol",
            "exchange",
            "security_type",
            "listing_status",
            "currency",
            "country",
            "primary_listing",
            "asset_class",
        ]

    for f in mandatory_fields:
        if f not in vals or vals[f] is None:
            raise ValueError(
                f"Cannot construct SourceSecurity without explicit, validated value for '{f}'. "
                f"Defaulting is strictly prohibited."
            )
        if isinstance(vals[f], str) and vals[f].strip() == "":
            raise ValueError(
                f"Field '{f}' contains empty string; cannot construct SourceSecurity."
            )

    # Temporal event fields must be explicitly validated (either date string or explicit None for 'no event')
    temporal_event_fields = [
        "listing_date",
        "delisting_date",
        "trading_suspension_date",
    ]
    for tf in temporal_event_fields:
        if tf not in vals:
            raise ValueError(
                f"Cannot construct SourceSecurity without validated value for '{tf}'. "
                f"Field was omitted from input record or outside validated requirements. "
                f"Explicit null (None) required for absence of event."
            )

    # Canonical incorporation country handling
    inc_country = vals.get("incorporation_country") or vals.get("country_of_incorporation")

    if is_listing_profile:
        # Explicit listing jurisdiction profile:
        # country is not assumed from listing_country; missing legacy country remains None
        # primary_listing is NOT substituted by selected_us_assessment_listing; defaults to False if unevidenced
        return SourceSecurity(
            security_id=str(vals["security_id"]),
            symbol=str(vals["symbol"]),
            exchange=str(vals["exchange"]),
            security_type=str(vals["security_type"]),
            listing_status=str(vals["listing_status"]),
            currency=str(vals["currency"]),
            country=str(vals["country"]) if "country" in vals and vals["country"] is not None else None,
            primary_listing=bool(vals["primary_listing"]) if "primary_listing" in vals and vals["primary_listing"] is not None else False,
            asset_class=str(vals["asset_class"]),
            listing_date=vals["listing_date"],
            delisting_date=vals["delisting_date"],
            trading_suspension_date=vals["trading_suspension_date"],
            listing_country=str(vals["listing_country"]),
            incorporation_country=str(inc_country) if inc_country is not None else None,
            selected_us_assessment_listing=bool(vals["selected_us_assessment_listing"]),
        )
    else:
        return SourceSecurity(
            security_id=str(vals["security_id"]),
            symbol=str(vals["symbol"]),
            exchange=str(vals["exchange"]),
            security_type=str(vals["security_type"]),
            listing_status=str(vals["listing_status"]),
            currency=str(vals["currency"]),
            country=str(vals["country"]),
            primary_listing=bool(vals["primary_listing"]),
            asset_class=str(vals["asset_class"]),
            listing_date=vals["listing_date"],
            delisting_date=vals["delisting_date"],
            trading_suspension_date=vals["trading_suspension_date"],
            listing_country=str(vals["listing_country"]) if "listing_country" in vals and vals["listing_country"] is not None else None,
            incorporation_country=str(inc_country) if inc_country is not None else None,
            selected_us_assessment_listing=(
                bool(vals["selected_us_assessment_listing"])
                if "selected_us_assessment_listing" in vals and vals["selected_us_assessment_listing"] is not None
                else None
            ),
        )
