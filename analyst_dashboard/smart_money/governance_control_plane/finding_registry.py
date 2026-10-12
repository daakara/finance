"""
Finding Registry Subsystem
ARX Terminal — Smart Money

Maintains the authoritative projection of architecture and governance findings.
Includes the canonical Section 46 known architecture findings with 7-layer reclassification.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib

from .canonicalization import sha256_canonical, domain_normalize


class FindingStatus(str, Enum):
    OPEN = "OPEN"
    ACTIVE = "ACTIVE"
    IN_REMEDIATION = "IN_REMEDIATION"
    REMEDIATED = "REMEDIATED"
    VERIFIED_RESOLVED = "VERIFIED_RESOLVED"
    WAIVED = "WAIVED"
    DISPROVED = "DISPROVED"
    CONTESTED = "CONTESTED"
    STALE = "STALE"


@dataclass
class Finding:
    finding_id: str
    title: str
    severity: str
    lifecycle_status: FindingStatus
    evidence_status: str
    owner: str
    resolution_authority: str
    verification_authority: str
    derived_blocks: List[str]
    reclassified_layer: str = "ARCHITECTURE_DEFECT"
    defect_class: str = "GENERIC_DEFECT"
    assertion_status: str = "ASSERTED"
    admission_status: str = "ADMITTED"  # ASSERTED_PENDING_REVIEW | ADMITTED | DUPLICATE | SUPERSEDED | REJECTED
    duplicate_of: Optional[str] = None
    superseded_by: Optional[str] = None
    architecture_contract_complete: bool = False
    implementation_present: bool = False
    external_evidence_required: bool = False
    qualification_required: bool = False
    classification_rationale: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    resolution_criteria: str = ""
    resolution_evidence: Optional[str] = None
    verification_evidence: Optional[str] = None
    waiver_expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lifecycle_status"] = self.lifecycle_status.value
        d.pop("assertion_status", None)
        d.pop("admission_status", None)
        d.pop("duplicate_of", None)
        d.pop("superseded_by", None)
        return d

    def to_admission_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lifecycle_status"] = self.lifecycle_status.value
        return d

    @property
    def original_subject(self) -> str:
        return FINDING_ORIGINAL_SUBJECTS.get(self.finding_id, self.title)

    @property
    def original_subject_normalized(self) -> str:
        return self.original_subject.strip().lower()

    @property
    def original_subject_hash(self) -> str:
        norm = self.original_subject_normalized
        if not norm:
            return ""
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @property
    def domain_scope(self) -> str:
        return FINDING_DOMAIN_SCOPES.get(self.finding_id, "OTHER")

    @property
    def defect_layer(self) -> str:
        return self.reclassified_layer

    def to_identity_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "original_subject": self.original_subject,
            "original_subject_normalized": self.original_subject_normalized,
            "original_subject_hash": self.original_subject_hash,
            "current_classification": {
                "domain_scope": self.domain_scope,
                "defect_layer": self.defect_layer,
            },
            "assertion_status": self.assertion_status,
            "admission_status": self.admission_status,
            "duplicate_of": self.duplicate_of,
            "superseded_by": self.superseded_by,
            "current_status": self.lifecycle_status.value,
            "current_scope": "PROVISIONING" if "PROVISIONING" in self.derived_blocks else "GENERAL",
            "evidence_refs": self.evidence_refs,
            "identity_preserved": True,
        }

    @property
    def is_blocking(self) -> bool:
        # Only admitted findings can block
        if self.admission_status != "ADMITTED":
            return False
        if self.lifecycle_status in [
            FindingStatus.VERIFIED_RESOLVED,
            FindingStatus.DISPROVED,
        ]:
            return False
        if self.lifecycle_status == FindingStatus.WAIVED:
            # If waived and not expired, non-blocking
            return False
        return self.severity in ["CRITICAL", "HIGH", "BLOCKING"]


    @property
    def blocks(self) -> Dict[str, bool]:
        arch_block = (self.reclassified_layer == "ARCHITECTURE_DEFECT" and self.is_blocking)
        prov_block = "PROVISIONING" in self.derived_blocks
        return {
            "architecture_closure": arch_block,
            "provisioning": prov_block,
            "theorem_A": "THEOREM_A" in self.derived_blocks,
            "theorem_B": "THEOREM_B" in self.derived_blocks,
            "theorem_C": "THEOREM_C" in self.derived_blocks,
            "theorem_D": "THEOREM_D" in self.derived_blocks,
            "theorem_E": "THEOREM_E" in self.derived_blocks,
            "natural_pre_arm": "NATURAL_PRE_ARM" in self.derived_blocks,
        }


FINDING_ORIGINAL_SUBJECTS: Dict[str, str] = {
    "FND-ARCH-001": "Candidate 004 freeze-file SHA-256 was conflated with Candidate 004 semantic-closure hash.",
    "FND-ARCH-002": "Genesis release provenance was unsupported or not derived from authoritative evidence.",
    "FND-ARCH-003": "Timestamp-authority provenance was claimed without adequate authoritative evidence.",
    "FND-ARCH-004": "Zero-event RPO requirement conflicted with asynchronous replication semantics.",
    "FND-ARCH-005": "Evidence-custody contract conflated provider-specific storage semantics.",
    "FND-ARCH-006": "OIDC identity binding was insufficiently scoped or incomplete.",
    "FND-ARCH-007": "Claimed-but-uncommitted Gate3 state lacked complete durable recovery semantics.",
    "FND-ARCH-008": "Gate3 claim contention was incorrectly mapped to semantic evidence quarantine.",
    "FND-ARCH-009": "Transient evidence-fetch/server failures were over-classified as terminal failures.",
    "FND-ARCH-010": "Stale fencing-token rejection was over-classified as a system terminal hold rather than normal stale-writer rejection.",
    "FND-ARCH-011": "Stored and derived denominator predicates were not proven logically equivalent.",
    "FND-ARCH-012": "Git commit SHA truncation in Genesis Manifest.",
    "FND-ARCH-013": "Gate-3 lease-expiry recovery behavior.",
    "FND-ARCH-014": "Transaction-log divergence from consensus.",
    "FND-ARCH-015": "Candidate-freeze predicate coupling with denominator evaluation oracle.",
}

FINDING_DOMAIN_SCOPES: Dict[str, str] = {
    "FND-ARCH-001": "DENOMINATOR",
    "FND-ARCH-002": "ROOT_OF_TRUST",
    "FND-ARCH-003": "TIMESTAMP_ORDERING",
    "FND-ARCH-004": "LEDGER_CONSENSUS",
    "FND-ARCH-005": "EVIDENCE_CUSTODY",
    "FND-ARCH-006": "GATE3_AUTHORITY",
    "FND-ARCH-007": "GATE3_AUTHORITY",
    "FND-ARCH-008": "GATE3_AUTHORITY",
    "FND-ARCH-009": "OBSERVATION_ENGINE",
    "FND-ARCH-010": "LEDGER_CONSENSUS",
    "FND-ARCH-011": "DENOMINATOR",
    "FND-ARCH-012": "ROOT_OF_TRUST",
    "FND-ARCH-013": "GATE3_AUTHORITY",
    "FND-ARCH-014": "LEDGER_CONSENSUS",
    "FND-ARCH-015": "DENOMINATOR",
}


def get_canonical_known_architecture_findings() -> List[Finding]:
    """
    Ingests the 11 known architecture findings mandated by Section 46 & 13.
    Reclassified according to the 7 allowed layers.
    """
    return [
        Finding(
            finding_id="FND-ARCH-001",
            title="semantic hash identity collision",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED_VULNERABILITY",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_A", "THEOREM_B"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="HASH_CANONICALIZATION_AMBIGUITY",
            architecture_contract_complete=False,
            implementation_present=True,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Semantic hash identity collisions in domain collection ordering remain an unresolved architecture defect across external substrate boundaries.",
            evidence_refs=["urn:arx:evidence:audit:hash-collision:001"],
            resolution_criteria="Universal adoption of ARX-GOV-NORMALIZE-V1 + RFC 8785 JCS contract across all external substrate data structures.",
        ),
        Finding(
            finding_id="FND-ARCH-002",
            title="genesis release provenance",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNVERIFIED_PROVENANCE",
            owner="urn:arx:identity:security-officer-alpha:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_A"],
            reclassified_layer="EXTERNAL_EVIDENCE_GAP",
            defect_class="MISSING_ATTESTATION",
            architecture_contract_complete=True,
            implementation_present=True,
            external_evidence_required=True,
            qualification_required=True,
            classification_rationale="The release trust architecture is fully specified; absence of external Sigstore bundle attestation is an external evidence gap, not an architecture defect.",
            evidence_refs=["urn:arx:evidence:audit:release-provenance:001"],
            resolution_criteria="Generate and verify cryptographic Sigstore bundle attesting Candidate 004 commit SHA.",
        ),
        Finding(
            finding_id="FND-ARCH-003",
            title="timestamp authority provenance",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNANCHORED_TIMESTAMPS",
            owner="urn:arx:identity:governance-lead-bravo:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_B", "THEOREM_D"],
            reclassified_layer="PROVIDER_SELECTION_PREREQUISITE",
            defect_class="UNSELECTED_EXTERNAL_PROVIDER",
            architecture_contract_complete=True,
            implementation_present=True,
            external_evidence_required=True,
            qualification_required=True,
            classification_rationale="RFC 3161 timestamping interface contract is frozen; selecting and configuring a specific external TSA/Rekor provider is a provisioning prerequisite.",
            evidence_refs=["urn:arx:evidence:audit:tsa-provenance:001"],
            resolution_criteria="Operator selection and endpoint configuration of RFC 3161 compliant timestamp authority.",
        ),
        Finding(
            finding_id="FND-ARCH-004",
            title="RPO vs asynchronous replication contradiction",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED_CONTRADICTION",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="REPLICATION_RPO_CONTRADICTION",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Normative requirement of RPO=0 events directly contradicts asynchronous multi-region replication design without synchronous quorum commits.",
            evidence_refs=["urn:arx:evidence:audit:rpo-async-contradiction:001"],
            resolution_criteria="Formalize synchronous multi-region commit protocol or revise RPO metric to match verified durable acknowledgment bounds.",
        ),
        Finding(
            finding_id="FND-ARCH-005",
            title="provider-specific custody semantics",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="LEAKED_ABSTRACTION",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_D"],
            reclassified_layer="PROVIDER_SELECTION_PREREQUISITE",
            defect_class="UNENCAPSULATED_STORAGE_PRIMITIVE",
            architecture_contract_complete=True,
            implementation_present=True,
            external_evidence_required=True,
            qualification_required=True,
            classification_rationale="Abstract WORM evidence custody interface is defined; mapping specific S3 Object Lock vs GCS Bucket Lock primitives is an operator provider selection prerequisite.",
            evidence_refs=["urn:arx:evidence:audit:custody-semantics:001"],
            resolution_criteria="Provision concrete WORM storage target and bind provider-neutral custody adapter.",
        ),
        Finding(
            finding_id="FND-ARCH-006",
            title="OIDC claim completeness",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="INCOMPLETE_CLAIMS",
            owner="urn:arx:identity:security-officer-alpha:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_B"],
            reclassified_layer="IMPLEMENTATION_PREREQUISITE",
            defect_class="MISSING_VERIFICATION_ASSERTIONS",
            architecture_contract_complete=True,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="OIDC claim verification requirements are specified; adding checks for repository_id, runner_environment, and ref is a code implementation task in qualification scripts.",
            evidence_refs=["urn:arx:evidence:audit:oidc-claims:001"],
            resolution_criteria="Implement explicit assertion of repository_id, runner_environment, and ref claims in qualify_sigstore_oidc.py.",
        ),
        Finding(
            finding_id="FND-ARCH-007",
            title="claimed-but-uncommitted Gate3 recovery semantics",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNRESOLVED_RECOVERY_GAP",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_B", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="CRASH_RECOVERY_AMBIGUITY",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Lack of deterministic protocol for reclaiming Gate-3 authorization claimed by crashed runner before committing is an architecture defect.",
            evidence_refs=["urn:arx:evidence:audit:gate3-recovery:001"],
            resolution_criteria="Specify two-phase commit or transactional lock-expiration protocol for uncommitted Gate-3 claims.",
        ),
        Finding(
            finding_id="FND-ARCH-008",
            title="claim contention vs quarantine semantics",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="AMBIGUOUS_LIFECYCLE",
            owner="urn:arx:identity:governance-lead-bravo:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="STATE_MACHINE_AMBIGUITY",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Concurrent claim contention semantics (transient retry) vs quarantine semantics (permanent invalidation) lack distinct state definitions.",
            evidence_refs=["urn:arx:evidence:audit:contention-quarantine:001"],
            resolution_criteria="Disambiguate CLAIM_CONTENTION (retryable) from EVENT_QUARANTINE (non-retryable terminal state) in finalization ledger.",
        ),
        Finding(
            finding_id="FND-ARCH-009",
            title="transient evidence fetch classification",
            severity="MEDIUM",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNCLASSIFIED_FETCHES",
            owner="urn:arx:identity:governance-lead-bravo:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_D"],
            reclassified_layer="POLICY_DEFECT",
            defect_class="ERROR_CLASSIFICATION_DEFECT",
            architecture_contract_complete=True,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=False,
            classification_rationale="SEC EDGAR network timeout classification is an operational policy defect in error taxonomy, not an architectural defect.",
            evidence_refs=["urn:arx:evidence:audit:transient-fetches:001"],
            resolution_criteria="Update policy error classification rule to distinguish transient network 5xx timeouts from corrupted payload bytes.",
        ),
        Finding(
            finding_id="FND-ARCH-010",
            title="stale fencing-token handling",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNCHECKED_TOKENS",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="CONCURRENCY_FENCING_DEFECT",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="State store lacks monotonic fencing token verification, creating risk of split-brain write by delayed zombie coordinator.",
            evidence_refs=["urn:arx:evidence:audit:fencing-tokens:001"],
            resolution_criteria="Implement strict monotonic fencing token check on every coordinator state transition and anchor attempt.",
        ),
        Finding(
            finding_id="FND-ARCH-011",
            title="denominator predicate alignment",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="PREDICATE_DRIFT",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "NATURAL_PRE_ARM"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="SEMANTIC_INVARIANT_MISALIGNMENT",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Pre-arm admission predicate and denominator increment condition have semantic drift in timing and signature validation requirements.",
            evidence_refs=["urn:arx:evidence:audit:denominator-alignment:001"],
            resolution_criteria="Align pre-arm admission predicate mathematically with denominator increment condition in passive capture contract.",
        ),
    ]


def get_additional_registered_findings() -> List[Finding]:
    """
    Returns the additional registered findings FND-ARCH-012 through FND-ARCH-015
    with their formal admission dispositions.
    """
    return [
        Finding(
            finding_id="FND-ARCH-012",
            title="git commit sha truncation",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED_VULNERABILITY",
            owner="urn:arx:identity:security-officer-alpha:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_A"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="SHA_TRUNCATION_RISK",
            assertion_status="ASSERTED",
            admission_status="ADMITTED",
            architecture_contract_complete=False,
            implementation_present=True,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Git commit SHA truncation in Genesis Manifest introduces hash collision and ambiguity risk.",
            evidence_refs=["urn:arx:evidence:audit:git-sha-truncation:001"],
            resolution_criteria="Ensure all git commit SHAs in Genesis Manifest and root contracts are full 40-character hexadecimal strings.",
        ),
        Finding(
            finding_id="FND-ARCH-013",
            title="gate-3 lease expiry recovery behavior",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="DUPLICATE_ASSERTION",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_B", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="CRASH_RECOVERY_AMBIGUITY",
            assertion_status="ASSERTED",
            admission_status="DUPLICATE",
            duplicate_of="FND-ARCH-007",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Duplicate of FND-ARCH-007 (claimed-but-uncommitted Gate3 recovery semantics).",
            evidence_refs=["urn:arx:evidence:audit:lease-expiry-recovery:001"],
            resolution_criteria="Resolved via FND-ARCH-007 resolution.",
        ),
        Finding(
            finding_id="FND-ARCH-014",
            title="transaction log divergence from consensus",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED_VULNERABILITY",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "THEOREM_C"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="REPLICATION_DIVERGENCE",
            assertion_status="ASSERTED",
            admission_status="ADMITTED",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Transaction log divergence from consensus during partition recovery represents an unresolved architecture defect.",
            evidence_refs=["urn:arx:evidence:audit:transaction-log-divergence:001"],
            resolution_criteria="Formalize consensus log reconciliation and verify anti-entropy repair protocol.",
        ),
        Finding(
            finding_id="FND-ARCH-015",
            title="candidate-freeze predicate coupling with denominator evaluation oracle",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="DUPLICATE_ASSERTION",
            owner="urn:arx:identity:architecture-lead-charlie:v1",
            resolution_authority="urn:arx:authority:bootstrap-council:v1",
            verification_authority="urn:arx:identity:independent-auditor-delta:v1",
            derived_blocks=["PROVISIONING", "NATURAL_PRE_ARM"],
            reclassified_layer="ARCHITECTURE_DEFECT",
            defect_class="SEMANTIC_INVARIANT_MISALIGNMENT",
            assertion_status="ASSERTED",
            admission_status="DUPLICATE",
            duplicate_of="FND-ARCH-011",
            architecture_contract_complete=False,
            implementation_present=False,
            external_evidence_required=False,
            qualification_required=True,
            classification_rationale="Duplicate of FND-ARCH-011 (denominator predicate alignment).",
            evidence_refs=["urn:arx:evidence:audit:predicate-coupling:001"],
            resolution_criteria="Resolved via FND-ARCH-011 resolution.",
        ),
    ]


def get_all_registered_findings() -> List[Finding]:
    """
    Returns all 15 registered findings across the universe (FND-ARCH-001 through FND-ARCH-015).
    """
    return get_canonical_known_architecture_findings() + get_additional_registered_findings()


def get_canonical_admitted_findings() -> List[Finding]:
    """
    Returns only admitted findings (13 items, excluding duplicates/rejected).
    """
    return [f for f in get_all_registered_findings() if f.admission_status == "ADMITTED"]


def create_canonical_finding_registry() -> "FindingRegistry":
    """
    Factory creating a FindingRegistry populated with all 15 registered findings.
    """
    return FindingRegistry(get_all_registered_findings())


class FindingRegistry:
    """
    Authoritative Finding Registry. Projections are materialized deterministically
    from ledger events.
    """

    def __init__(self, findings: Optional[List[Finding]] = None):
        self._findings: Dict[str, Finding] = {}
        if findings:
            for f in findings:
                self._findings[f.finding_id] = f

    @property
    def findings(self) -> List[Finding]:
        return sorted(self._findings.values(), key=lambda x: x.finding_id)

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        return self._findings.get(finding_id)

    def add_finding(self, finding: Finding) -> None:
        self._findings[finding.finding_id] = finding

    def update_status(self, finding_id: str, new_status: FindingStatus) -> None:
        if finding_id in self._findings:
            self._findings[finding_id].lifecycle_status = new_status

    def compute_registry_hash(self) -> str:
        findings_payload = [f.to_dict() for f in self.findings]
        return sha256_canonical(findings_payload)

    def get_active_blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.is_blocking]

    def get_layer_counts(self) -> Dict[str, int]:
        counts = {
            "ARCHITECTURE_DEFECT": 0,
            "POLICY_DEFECT": 0,
            "PROVIDER_SELECTION_PREREQUISITE": 0,
            "IMPLEMENTATION_PREREQUISITE": 0,
            "EXTERNAL_EVIDENCE_GAP": 0,
            "QUALIFICATION_PREREQUISITE": 0,
            "DOCUMENTATION_DEFECT": 0,
        }
        for f in self.findings:
            if f.reclassified_layer in counts:
                counts[f.reclassified_layer] += 1
        return counts

    def get_unresolved_architecture_defects(self) -> List[Finding]:
        return [
            f for f in self.findings
            if f.reclassified_layer == "ARCHITECTURE_DEFECT"
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED]
        ]

    def get_dispositions(self) -> Dict[str, int]:
        total = len(self.findings)
        open_count = sum(1 for f in self.findings if f.lifecycle_status in [FindingStatus.OPEN, FindingStatus.ACTIVE, FindingStatus.IN_REMEDIATION])
        contested = sum(1 for f in self.findings if f.lifecycle_status == FindingStatus.CONTESTED)
        stale = sum(1 for f in self.findings if f.lifecycle_status == FindingStatus.STALE)
        verified_resolved = sum(1 for f in self.findings if f.lifecycle_status == FindingStatus.VERIFIED_RESOLVED)
        disproved = sum(1 for f in self.findings if f.lifecycle_status == FindingStatus.DISPROVED)
        return {
            "total": total,
            "open": open_count,
            "contested": contested,
            "stale": stale,
            "verified_resolved": verified_resolved,
            "disproved": disproved,
        }
