"""
SEC Source Authority Qualification & PDS Closure Engine (Epoch 001)

Formalizes:
1. Multi-dimensional Source Authority model (Population, Interval Closure, Public Availability, Correction, Finality)
2. PDS Daily EOD Closed-Interval Theorem & Exact Set Reconciliation
3. PDS Transport Identity vs Filing Domain Identity separation
4. Post-Acceptance Correction (PAC) Lineage & Non-Text Invariance
5. Public Availability Observation Channel & Timing Separation
6. Decision-Input Availability Contract (SMART_MONEY_OPEN_MARKET_PURCHASE_V1)
7. SEC Evidence Bundle V1 Schema & Append-Only Hash-Chaining
8. Legacy Timing Semantic Reconciliation Metadata
"""

import re
import json
import hashlib
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash


# =====================================================================
# 1. Source Authority Dimensions (Section 5)
# =====================================================================

class PopulationAuthority(str, Enum):
    OBSERVATIONAL = "OBSERVATIONAL"
    BOUNDED_STABLE = "BOUNDED_STABLE"
    AUTHORITATIVE_AS_OF_SNAPSHOT = "AUTHORITATIVE_AS_OF_SNAPSHOT"
    AUTHORITATIVE_CLOSED_INTERVAL = "AUTHORITATIVE_CLOSED_INTERVAL"


class IntervalClosureAuthority(str, Enum):
    NONE = "NONE"
    STABILITY_BASED = "STABILITY_BASED"
    SOURCE_NATIVE_TERMINAL_MARKER = "SOURCE_NATIVE_TERMINAL_MARKER"


class PublicAvailabilityTimeAuthority(str, Enum):
    UNKNOWN = "UNKNOWN"
    ACCEPTANCE_LOWER_BOUND_ONLY = "ACCEPTANCE_LOWER_BOUND_ONLY"
    DIRECT_METADATA_PUBLIC_OBSERVATION = "DIRECT_METADATA_PUBLIC_OBSERVATION"
    DIRECT_CONTENT_PUBLIC_OBSERVATION = "DIRECT_CONTENT_PUBLIC_OBSERVATION"
    SOURCE_NATIVE_EXACT = "SOURCE_NATIVE_EXACT"


class CorrectionAuthority(str, Enum):
    NONE = "NONE"
    SOURCE_SNAPSHOT_AS_OF = "SOURCE_SNAPSHOT_AS_OF"
    PDS_PAC_AWARE = "PDS_PAC_AWARE"
    RECONCILED_SOURCE_HISTORY = "RECONCILED_SOURCE_HISTORY"


class HistoricalFinality(str, Enum):
    NO = "NO"
    SOURCE_DEFINED = "SOURCE_DEFINED"


# Hard Invariants
SOURCE_FORMAT_VERSION = "PDS_FORMAT_2025_V2.8"
SOURCE_AUTHORITY_VERSION = "SEC_AUTH_EPOCH_001_V1.0"
HOLD_TIME_CREATES_SOURCE_AUTHORITY = False
SOURCE_REPLAY_SUCCESS_CREATES_EXTERNAL_SOURCE_GUARANTEE = False
PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT = False
SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY = False


# =====================================================================
# 2. PDS Source-Day & Transport Identity (Section 8, 11)
# =====================================================================

@dataclass(frozen=True)
class PdsBusinessDayIdentity:
    """
    Represents an official EDGAR business day in America/New_York.
    EDGAR dissemination runs 06:00:00 to ~22:00:00 ET.
    """
    business_day: str  # YYYY-MM-DD
    timezone: str = "America/New_York"
    window_start: str = "06:00:00"
    window_end: str = "22:00:00"
    terminal_artifact_pattern: str = r"^dissem[12]?_\d{2}-\d{2}-\d{4}\.EOD$"

    def get_expected_eod_filename(self, prefix: str = "dissem2_") -> str:
        parts = self.business_day.split("-")
        if len(parts) == 3:
            yyyy, mm, dd = parts
            return f"{prefix}{mm}-{dd}-{yyyy}.EOD"
        return f"{prefix}{self.business_day}.EOD"


def classify_pds_artifact_type(filename: str) -> str:
    """Classifies PDS filename extension into contract-defined artifact classes."""
    lower = filename.lower()
    if lower.endswith(".eod"):
        return "EOD"
    elif lower.endswith(".dissem"):
        return "DISSEM"
    elif lower.endswith(".paper"):
        return "PAPER"
    elif lower.endswith(".pr"):
        return "PR"
    elif lower.endswith(".pc"):
        return "PC"
    return "UNKNOWN"


def extract_accession_from_pds_filename(filename: str) -> Optional[str]:
    """
    Extracts SEC accession number from standard PDS filename if derivable.
    Standard pattern: [timestamp]-[accession].[ext] e.g. 20261009142310-0000320193-26-000100.dissem
    """
    m = re.search(r"(\d{10}-\d{2}-\d{6})", filename)
    if m:
        return m.group(1)
    return None


# =====================================================================
# 3. EOD Expected-Set Parser & Diagnostics (Section 10)
# =====================================================================

@dataclass
class PdsEodManifestEntry:
    raw_line_number: int
    raw_line: str
    exact_filename: str
    artifact_type: str
    parsed_source_day: str
    accession_if_derivable: Optional[str]


@dataclass
class PdsEodManifest:
    business_day: str
    eod_filename: str
    raw_bytes_sha256: str
    entries: List[PdsEodManifestEntry]
    raw_line_count: int
    valid_entry_count: int
    duplicate_filename_count: int
    malformed_entry_count: int
    unknown_entry_type_count: int

    @classmethod
    def parse_raw_eod(cls, raw_content: str, business_day: str, eod_filename: str) -> "PdsEodManifest":
        raw_bytes = raw_content.encode("utf-8")
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()

        lines = raw_content.splitlines()
        entries: List[PdsEodManifestEntry] = []
        seen_filenames: Set[str] = set()

        duplicate_count = 0
        malformed_count = 0
        unknown_type_count = 0

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            # In standard PDS EOD files, each line contains the exact disseminated filename
            fname = stripped
            atype = classify_pds_artifact_type(fname)

            if atype == "UNKNOWN":
                unknown_type_count += 1
            if len(fname) < 5 or " " in fname:
                malformed_count += 1

            if fname in seen_filenames:
                duplicate_count += 1
            seen_filenames.add(fname)

            acc = extract_accession_from_pds_filename(fname)
            entry = PdsEodManifestEntry(
                raw_line_number=line_num,
                raw_line=line,
                exact_filename=fname,
                artifact_type=atype,
                parsed_source_day=business_day,
                accession_if_derivable=acc,
            )
            entries.append(entry)

        return cls(
            business_day=business_day,
            eod_filename=eod_filename,
            raw_bytes_sha256=raw_sha,
            entries=entries,
            raw_line_count=len(lines),
            valid_entry_count=len(entries),
            duplicate_filename_count=duplicate_count,
            malformed_entry_count=malformed_count,
            unknown_entry_type_count=unknown_type_count,
        )

    def compute_expected_set_hash(self) -> str:
        """Deterministic hash of sorted unique expected filenames."""
        sorted_filenames = sorted([e.exact_filename for e in self.entries])
        return compute_canonical_hash(sorted_filenames)


# =====================================================================
# 4. Accounted PDS Set & Recovery Mechanisms (Section 12, 13, 15, 16)
# =====================================================================

class PdsReceiptMethod(str, Enum):
    LIVE_PDS = "LIVE_PDS"
    PDS_RETRANSMISSION = "PDS_RETRANSMISSION"
    PDS_STAGING_RECOVERY = "PDS_STAGING_RECOVERY"
    PDS_FTP_RECOVERY = "PDS_FTP_RECOVERY"
    PDS_HELPDESK_RECOVERY = "PDS_HELPDESK_RECOVERY"


@dataclass
class PdsAccountedArtifact:
    pds_filename: str
    source_business_day: str
    receipt_day: str
    received_at: str
    receipt_method: PdsReceiptMethod
    raw_bytes_sha256: str
    raw_bytes_retained: bool
    transport_provenance: Dict[str, Any]


@dataclass
class PdsAccountedSet:
    business_day: str
    artifacts: List[PdsAccountedArtifact]

    def compute_accounted_set_hash(self) -> str:
        sorted_fnames = sorted([a.pds_filename for a in self.artifacts if a.source_business_day == self.business_day])
        return compute_canonical_hash(sorted_fnames)

    def check_content_conflicts(self) -> Tuple[int, List[str]]:
        """
        Detects SAME_FILENAME_DIFFERENT_BYTES (Section 15).
        Returns count of conflicting filenames and list of conflicted filenames.
        """
        hashes_by_name: Dict[str, Set[str]] = {}
        for a in self.artifacts:
            hashes_by_name.setdefault(a.pds_filename, set()).add(a.raw_bytes_sha256)

        conflicts = [fname for fname, hashes in hashes_by_name.items() if len(hashes) > 1]
        return len(conflicts), conflicts

    def check_cross_day_retransmissions(self) -> Tuple[int, List[str]]:
        """
        Detects unclassified cross-day retransmissions (Section 13).
        Artifacts received on Day D+1 must be properly attributed to source day D.
        """
        unclassified: List[str] = []
        for a in self.artifacts:
            if a.receipt_day != a.source_business_day:
                if a.receipt_method not in (
                    PdsReceiptMethod.PDS_RETRANSMISSION,
                    PdsReceiptMethod.PDS_STAGING_RECOVERY,
                    PdsReceiptMethod.PDS_FTP_RECOVERY,
                    PdsReceiptMethod.PDS_HELPDESK_RECOVERY,
                ):
                    unclassified.append(a.pds_filename)
        return len(unclassified), unclassified


# =====================================================================
# 5. Post-Acceptance Correction (PAC) Semantics (Section 17)
# =====================================================================

@dataclass
class PdsPacArtifact:
    pds_artifact_id: str  # filename with .pc extension
    parent_accession: str
    correction_relation: str  # HEADER_TAG_UPDATE, DELETION_NOTICE, FORM_TYPE_ALTERATION
    observed_at: str
    timestamp_tag: str  # exact value from SEC <TIMESTAMP>
    source_business_day: str
    does_not_modify_document_text: bool = True


# =====================================================================
# 6. PDS Daily EOD Source Closure Theorem (Section 18)
# =====================================================================

@dataclass
class PdsSourceClosureResult:
    can_close: bool
    population_authority: PopulationAuthority
    interval_closure_authority: IntervalClosureAuthority
    authority_scope: str
    historical_finality: HistoricalFinality
    eod_expected_count: int
    pds_accounted_count: int
    expected_minus_accounted_count: int
    accounted_minus_expected_count: int
    missing_pds_files: List[str]
    unexpected_pds_files: List[str]
    same_filename_content_conflicts: int
    unclassified_cross_day_retransmissions: int
    eod_expected_set_hash: str
    pds_accounted_set_hash: str
    failure_reasons: List[str]


class PdsSourceClosureTheorem:
    """
    Pure deterministic implementation of the Section 18 Source Closure Theorem.
    """

    @classmethod
    def evaluate(
        cls,
        manifest: PdsEodManifest,
        accounted: PdsAccountedSet,
        pds_source_contract_qualified: bool = True,
        pds_source_contract_version_bound: bool = True,
        pds_business_day_bound: bool = True,
        raw_eod_artifact_retained: bool = True,
        eod_is_source_native_terminal_marker: bool = True,
        offline_source_replay_pass: bool = True,
        independent_source_reconstruction_pass: bool = True,
    ) -> PdsSourceClosureResult:
        failures: List[str] = []

        if not pds_source_contract_qualified:
            failures.append("PDS_SOURCE_CONTRACT_NOT_QUALIFIED")
        if not pds_source_contract_version_bound:
            failures.append("PDS_SOURCE_CONTRACT_VERSION_NOT_BOUND")
        if not pds_business_day_bound:
            failures.append("PDS_BUSINESS_DAY_NOT_BOUND")
        if not raw_eod_artifact_retained:
            failures.append("RAW_EOD_ARTIFACT_NOT_RETAINED")
        if not eod_is_source_native_terminal_marker:
            failures.append("EOD_NOT_SOURCE_NATIVE_TERMINAL_MARKER")
        if not offline_source_replay_pass:
            failures.append("OFFLINE_SOURCE_REPLAY_FAILED")
        if not independent_source_reconstruction_pass:
            failures.append("INDEPENDENT_SOURCE_RECONSTRUCTION_FAILED")

        # Manifest parse diagnostics
        if manifest.malformed_entry_count > 0:
            failures.append(f"EOD_MALFORMED_ENTRY_COUNT_NON_ZERO: {manifest.malformed_entry_count}")
        if manifest.unknown_entry_type_count > 0:
            failures.append(f"EOD_UNKNOWN_ENTRY_TYPE_COUNT_NON_ZERO: {manifest.unknown_entry_type_count}")
        if manifest.duplicate_filename_count > 0:
            failures.append(f"EOD_DUPLICATE_FILENAME_COUNT_NON_ZERO: {manifest.duplicate_filename_count}")

        # Accounted set checks
        conflicts_count, conflicted_files = accounted.check_content_conflicts()
        if conflicts_count > 0:
            failures.append(f"SAME_FILENAME_DIFFERENT_BYTES: {conflicts_count} files ({conflicted_files[:3]})")

        unclass_count, unclass_files = accounted.check_cross_day_retransmissions()
        if unclass_count > 0:
            failures.append(f"UNCLASSIFIED_CROSS_DAY_RETRANSMISSIONS: {unclass_count} files")

        # Set Reconciliation
        expected_fnames = {e.exact_filename for e in manifest.entries}
        accounted_fnames = {a.pds_filename for a in accounted.artifacts if a.source_business_day == manifest.business_day}

        missing = sorted(list(expected_fnames - accounted_fnames))
        unexpected = sorted(list(accounted_fnames - expected_fnames))

        if missing:
            failures.append(f"EOD_EXPECTED_MINUS_ACCOUNTED_NON_ZERO: {len(missing)} missing files")
        if unexpected:
            failures.append(f"PDS_ACCOUNTED_MINUS_EOD_EXPECTED_NON_ZERO: {len(unexpected)} unexpected files")

        eod_exp_hash = manifest.compute_expected_set_hash()
        pds_acc_hash = accounted.compute_accounted_set_hash()

        if eod_exp_hash != pds_acc_hash:
            failures.append("EXPECTED_AND_ACCOUNTED_SET_HASH_MISMATCH")

        can_close = len(failures) == 0

        pop_auth = (
            PopulationAuthority.AUTHORITATIVE_CLOSED_INTERVAL
            if can_close
            else PopulationAuthority.OBSERVATIONAL
        )
        closure_auth = (
            IntervalClosureAuthority.SOURCE_NATIVE_TERMINAL_MARKER
            if can_close
            else IntervalClosureAuthority.NONE
        )

        return PdsSourceClosureResult(
            can_close=can_close,
            population_authority=pop_auth,
            interval_closure_authority=closure_auth,
            authority_scope="PDS_DISSEMINATION_POPULATION_FOR_ONE_EDGAR_BUSINESS_DAY",
            historical_finality=HistoricalFinality.NO,
            eod_expected_count=len(expected_fnames),
            pds_accounted_count=len(accounted_fnames),
            expected_minus_accounted_count=len(missing),
            accounted_minus_expected_count=len(unexpected),
            missing_pds_files=missing,
            unexpected_pds_files=unexpected,
            same_filename_content_conflicts=conflicts_count,
            unclassified_cross_day_retransmissions=unclass_count,
            eod_expected_set_hash=eod_exp_hash,
            pds_accounted_set_hash=pds_acc_hash,
            failure_reasons=failures,
        )


# =====================================================================
# 7. Scoped Domain Derivation Theorem (Section 19)
# =====================================================================

GOVERNED_UNIVERSE_BUILD_ID = "UB_SM_20261010_R1_19ef857fec1f"
GOVERNED_UNIVERSE_CONTENT_HASH = "ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c"
GOVERNED_REQUIRED_PARTITION_SET_HASH = "e8bfa161ef8ef747db32fb192ca45ee7eb373378fdb53a30a69bb8fdb118bc33"


@dataclass
class ScopedDerivationResult:
    is_complete: bool
    governed_scope_expected_count: int
    governed_scope_registered_count: int
    pac_unresolved_count: int
    missing_scope_accessions: List[str]
    failure_reasons: List[str]


class ScopedDomainDerivationTheorem:
    """
    Evaluates:
    Closed PDS Population -> PAC Application -> Form 4 / 4-A Filter -> Governed 106-CIK Filter.
    """

    @classmethod
    def evaluate(
        cls,
        pds_closure_result: PdsSourceClosureResult,
        pds_artifacts: List[Dict[str, Any]],
        governed_ciks: Set[str],
        unresolved_pacs: List[PdsPacArtifact],
        universe_build_id: str = GOVERNED_UNIVERSE_BUILD_ID,
        required_partition_set_hash: str = GOVERNED_REQUIRED_PARTITION_SET_HASH,
    ) -> ScopedDerivationResult:
        failures: List[str] = []

        if not pds_closure_result.can_close:
            failures.append("PDS_SOURCE_INTERVAL_NOT_CLOSED")

        if universe_build_id != GOVERNED_UNIVERSE_BUILD_ID:
            failures.append("UNIVERSE_BUILD_ID_MISMATCH")

        if required_partition_set_hash != GOVERNED_REQUIRED_PARTITION_SET_HASH:
            failures.append("REQUIRED_PARTITION_SET_HASH_MISMATCH")

        if unresolved_pacs:
            failures.append(f"PAC_UNRESOLVED: {len(unresolved_pacs)} unresolved PAC actions")

        # Derive Form 4 within governed 106 CIK scope
        expected_in_scope: Set[str] = set()
        registered_in_scope: Set[str] = set()

        for art in pds_artifacts:
            acc = art.get("accession_number")
            cik = art.get("issuer_cik", "").strip().zfill(10)
            form = art.get("form_type", "").strip()

            if cik in governed_ciks and form in ("4", "4/A"):
                if acc:
                    expected_in_scope.add(acc)
                    if art.get("is_registered", False):
                        registered_in_scope.add(acc)

        missing = sorted(list(expected_in_scope - registered_in_scope))
        if missing:
            failures.append(f"SCOPED_ACCESSIONS_UNREGISTERED: {len(missing)} accessions missing")

        is_comp = len(failures) == 0
        return ScopedDerivationResult(
            is_complete=is_comp,
            governed_scope_expected_count=len(expected_in_scope),
            governed_scope_registered_count=len(registered_in_scope),
            pac_unresolved_count=len(unresolved_pacs),
            missing_scope_accessions=missing,
            failure_reasons=failures,
        )


# =====================================================================
# 8. Public Observation & Timing Channel Separation (Section 20, 21, 22, 24, 25, 26)
# =====================================================================

class PublicEndpointClass(str, Enum):
    FILING_INDEX = "FILING_INDEX"
    COMPLETE_SUBMISSION_TEXT = "COMPLETE_SUBMISSION_TEXT"
    PRIMARY_FORM_DOCUMENT = "PRIMARY_FORM_DOCUMENT"
    OWNERSHIP_XML = "OWNERSHIP_XML"
    ATTACHMENT = "ATTACHMENT"
    RSS_FEED = "RSS_FEED"
    SUBMISSIONS_API = "SUBMISSIONS_API"


class PublicObservationResult(str, Enum):
    METADATA_PRESENT = "METADATA_PRESENT"
    CONTENT_PRESENT = "CONTENT_PRESENT"
    ENDPOINT_NOT_PRESENT = "ENDPOINT_NOT_PRESENT"
    TRANSPORT_INDETERMINATE = "TRANSPORT_INDETERMINATE"
    IDENTITY_INVALID = "IDENTITY_INVALID"


@dataclass
class PublicObservationRecord:
    observation_id: str
    accession_number: str
    endpoint_class: PublicEndpointClass
    url_or_source_id: str
    request_started_at: str
    response_completed_at: str
    clock_source: str
    clock_offset_ms: int
    clock_uncertainty_ms: int
    clock_status: str  # SYNCHRONIZED, DRIFT_DETECTED, UNKNOWN
    http_status: int
    raw_response_retained: bool
    raw_response_sha256: str
    observed_accession: Optional[str]
    observed_cik: Optional[str]
    observed_form: Optional[str]
    observation_result: PublicObservationResult
    bytes_received: int
    content_length_header: Optional[int] = None
    content_encoding: Optional[str] = None

    def get_conservative_knowledge_time(self) -> str:
        """
        Conservative knowledge time binds to response completion time, NOT request start time.
        """
        return self.response_completed_at


def classify_http_observation(
    status_code: int,
    bytes_received: int,
    content_length: Optional[int],
    accession_in_body: Optional[str],
    target_accession: str,
    is_content_endpoint: bool,
) -> PublicObservationResult:
    """
    Enforces Section 22 and 24 observation classification:
    403 / 429 / 5xx / timeout / network failure -> TRANSPORT_INDETERMINATE (never proof of absence).
    Truncated response -> TRANSPORT_INDETERMINATE (TRANSPORT_INCOMPLETE).
    Identity mismatch -> IDENTITY_INVALID.
    """
    if status_code in (403, 429, 500, 502, 503, 504, 0):
        return PublicObservationResult.TRANSPORT_INDETERMINATE

    if status_code == 404:
        return PublicObservationResult.ENDPOINT_NOT_PRESENT

    if status_code == 200:
        if content_length is not None and bytes_received < content_length:
            return PublicObservationResult.TRANSPORT_INDETERMINATE

        if accession_in_body and accession_in_body != target_accession:
            return PublicObservationResult.IDENTITY_INVALID

        return (
            PublicObservationResult.CONTENT_PRESENT
            if is_content_endpoint
            else PublicObservationResult.METADATA_PRESENT
        )

    return PublicObservationResult.TRANSPORT_INDETERMINATE


# =====================================================================
# 9. Decision-Input Availability Contract (Section 23)
# =====================================================================

DECISION_INPUT_CONTRACT_ID = "SMART_MONEY_OPEN_MARKET_PURCHASE_V1"

# Exact specification derived from actual Smart Money implementation (form4_parser.py)
DECISION_INPUT_CONTRACT_SPEC = {
    "contract_id": DECISION_INPUT_CONTRACT_ID,
    "version": "1.0.0",
    "description": "Deterministic decision-input dependency contract for Smart Money Open Market Purchases",
    "required_metadata_fields": [
        "issuerCik",
        "documentType",
        "periodOfReport",
    ],
    "required_reporting_owner_fields": [
        "rptOwnerCik",
        "rptOwnerName",
        "relationship",  # at least one of isDirector, isOfficer, isTenPercentOwner
    ],
    "required_transaction_fields": [
        "transactionDate",
        "transactionCode",  # must be 'P'
        "transactionAcquiredDisposedCode",  # must be 'A'
        "transactionShares",  # float > 0
        "transactionPricePerShare",  # float > 0
        "directOrIndirectOwnership",  # 'D' or 'I'
    ],
    "conditional_fields": {
        "officerTitle": "Required if isOfficer is True",
    },
    "footnote_dependencies": {
        "rule_10b5_1": "Resolved if footnoteId referenced",
        "weighted_average_price": "Resolved if footnoteId referenced",
    },
    "timing_rule": "FIRST_PROVEN_DECISION_INPUT_COMPLETE_AT = earliest qualifying response_completed_at",
}

DECISION_INPUT_CONTRACT_SHA256 = compute_canonical_hash(DECISION_INPUT_CONTRACT_SPEC)


@dataclass
class DecisionInputResult:
    contract_id: str
    contract_sha256: str
    is_decision_input_complete: bool
    missing_fields: List[str]
    first_proven_decision_input_complete_at: Optional[str]


class DecisionInputEvaluator:
    """
    Evaluates whether an observed SEC filing satisfies the deterministic
    SMART_MONEY_OPEN_MARKET_PURCHASE_V1 decision contract.
    """

    @classmethod
    def evaluate(
        cls,
        filing_data: Dict[str, Any],
        observation_record: Optional[PublicObservationRecord] = None,
    ) -> DecisionInputResult:
        missing: List[str] = []

        # 1. Metadata
        if not filing_data.get("issuer_cik"):
            missing.append("issuerCik")
        doc_type = filing_data.get("document_type", "")
        if doc_type not in ("4", "4/A"):
            missing.append("documentType")
        if not filing_data.get("filing_date"):
            missing.append("periodOfReport")

        # 2. Reporting Owner
        if not filing_data.get("reporting_owner_cik"):
            missing.append("rptOwnerCik")
        if not filing_data.get("reporting_owner_name"):
            missing.append("rptOwnerName")

        has_rel = (
            filing_data.get("is_director") is not None
            or filing_data.get("is_officer") is not None
            or filing_data.get("is_ten_percent_owner") is not None
        )
        if not has_rel:
            missing.append("relationship")

        # 3. Transactions
        txs = filing_data.get("transactions", [])
        if not txs:
            missing.append("transactions[non_empty]")
        else:
            for i, tx in enumerate(txs):
                if not tx.get("transaction_date"):
                    missing.append(f"transactions[{i}].transactionDate")
                if tx.get("transaction_code") != "P":
                    missing.append(f"transactions[{i}].transactionCode=='P'")
                if tx.get("acquired_disposed") != "A":
                    missing.append(f"transactions[{i}].acquiredDisposed=='A'")
                if (tx.get("shares") or 0.0) <= 0.0:
                    missing.append(f"transactions[{i}].shares>0")
                if (tx.get("price_per_share") or 0.0) <= 0.0:
                    missing.append(f"transactions[{i}].price_per_share>0")
                if tx.get("ownership_nature") not in ("D", "I", "DIRECT", "INDIRECT"):
                    missing.append(f"transactions[{i}].directOrIndirectOwnership")

        is_comp = len(missing) == 0
        comp_time = None
        if is_comp and observation_record:
            comp_time = observation_record.get_conservative_knowledge_time()

        return DecisionInputResult(
            contract_id=DECISION_INPUT_CONTRACT_ID,
            contract_sha256=DECISION_INPUT_CONTRACT_SHA256,
            is_decision_input_complete=is_comp,
            missing_fields=missing,
            first_proven_decision_input_complete_at=comp_time,
        )


# =====================================================================
# 10. Evidence-Bundle Schema & Append-Only Lineage (Section 28, 29)
# =====================================================================

BUNDLE_SCHEMA_VERSION = "SEC_EVIDENCE_BUNDLE_V1"
EVIDENCE_COMPATIBILITY_EPOCH = "EPOCH_001"


@dataclass
class SecEvidenceBundleV1:
    bundle_schema_version: str
    bundle_id: str
    bundle_sequence: int
    previous_bundle_id: Optional[str]
    previous_bundle_sha256: Optional[str]
    source_contract_id: str
    source_contract_version: str
    source_contract_sha256: str
    canonicalization_version: str
    canonicalization_sha256: str
    derivation_policy_version: str
    derivation_policy_sha256: str
    parser_version: str
    parser_git_sha: str
    evidence_compatibility_epoch: str
    business_day: str
    manifest_data: Dict[str, Any]
    bundle_sha256: str = ""

    def compute_bundle_hash(self) -> str:
        d = asdict(self)
        d.pop("bundle_sha256", None)
        return compute_canonical_hash(d)

    def seal(self) -> "SecEvidenceBundleV1":
        h = self.compute_bundle_hash()
        return SecEvidenceBundleV1(
            bundle_schema_version=self.bundle_schema_version,
            bundle_id=self.bundle_id,
            bundle_sequence=self.bundle_sequence,
            previous_bundle_id=self.previous_bundle_id,
            previous_bundle_sha256=self.previous_bundle_sha256,
            source_contract_id=self.source_contract_id,
            source_contract_version=self.source_contract_version,
            source_contract_sha256=self.source_contract_sha256,
            canonicalization_version=self.canonicalization_version,
            canonicalization_sha256=self.canonicalization_sha256,
            derivation_policy_version=self.derivation_policy_version,
            derivation_policy_sha256=self.derivation_policy_sha256,
            parser_version=self.parser_version,
            parser_git_sha=self.parser_git_sha,
            evidence_compatibility_epoch=self.evidence_compatibility_epoch,
            business_day=self.business_day,
            manifest_data=self.manifest_data,
            bundle_sha256=h,
        )


# =====================================================================
# 11. Legacy Timing Semantic Reconciliation Metadata (Section 27)
# =====================================================================

@dataclass(frozen=True)
class LegacyTimingSemanticReconciliation:
    legacy_field: str = "public_availability_at"
    legacy_value_source: str = "SEC_ACCEPTANCE_DATETIME"
    reconciled_semantic: str = "sec_acceptance_at"
    historical_first_public_available_at: str = "UNKNOWN_UNLESS_SEPARATELY_OBSERVED"
    legacy_rows_mutated: bool = False
    pit_affected_code_paths: Tuple[str, ...] = (
        "analyst_dashboard/smart_money/form4_models.py (SmartMoneyTransaction.__post_init__)",
        "analyst_dashboard/smart_money/form4_parser.py (Form4XmlParser.parse_filing)",
        "analyst_dashboard/smart_money/form4_ingestion.py (query_smart_money_transactions_as_of)",
        "analyst_dashboard/smart_money/incremental_engine.py (compute_source_max_timestamp)",
        "analyst_dashboard/smart_money/policy_bundle.py (PolicyBundleAuthority)",
    )
