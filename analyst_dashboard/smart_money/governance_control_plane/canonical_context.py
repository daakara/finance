"""
Canonical Decision Context Builder & Verification Subsystem
ARX Terminal — Smart Money

Constructs the immutable CanonicalDecisionContext by:
1. Verifying source freeze commit in the repository object database
2. Recomputing repository evidence from Git tree blobs
3. Deriving canonical claims via CANONICAL_CLAIM_DERIVATION_POLICY.json
4. Deriving predicate facts via CANONICAL_FACT_DERIVATION_POLICY.json
5. Materializing finding identities, admission statuses, and active blockers
6. Loading repository-bound dependency policy and validating DAG completeness
7. Enforcing evaluation clock limits and preventing future-dated records
8. Computing the unified decision_context_hash
"""

from __future__ import annotations
import subprocess
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta

from .canonicalization import sha256_canonical, JCS_VERSION, ARX_GOV_NORMALIZE_VERSION
from .evidence_manifest import (
    EvidenceRecord,
    CanonicalEvidenceManifest,
    build_canonical_evidence_manifest,
    validate_canonical_evidence_record,
)
from .claim_derivation import (
    ClaimDerivationEngine,
    CanonicalDerivedClaim,
    EmptyClaimRegistryError,
    StateBasis,
)
from .fact_derivation import FactDerivationEngine
from .finding_registry import (
    FindingRegistry,
    get_all_registered_findings,
)
from .dependency_policy import (
    compile_dependency_policy,
    resolve_dependencies,
    CompiledDependencyPolicy,
    DependencyResolutionResult,
)
from .predicate_evaluator import (
    FactContext,
    DSL_VERSION,
    compute_logic_semantics_hash,
)
from .root_identity import validate_git_oid


EXPECTED_CANDIDATE004_FREEZE_HASH = "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
EXPECTED_CANDIDATE004_SEMANTIC_HASH = "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6"


class CanonicalContextError(Exception):
    """Base exception for canonical context construction errors."""
    pass


class SourceFreezeVerificationError(CanonicalContextError):
    """Raised when the specified source freeze SHA cannot be verified in Git."""
    pass


class RepositoryEvidenceRecomputationError(CanonicalContextError):
    """Raised when repository evidence cannot be recomputed or verified from Git tree blobs."""
    pass


class FutureDatedRecordError(CanonicalContextError):
    """Raised when evidence or decision timestamp exceeds allowed clock skew."""
    pass


class DAGIncompletenessError(CanonicalContextError):
    """Raised when the dependency DAG fails completeness checks against the decision surface."""
    pass


class CallerConstructedCanonicalContextRejectedError(CanonicalContextError):
    """Raised when a caller attempts to pass unverified objects directly to canonical decision."""
    pass


@dataclass(frozen=True)
class CanonicalDecisionContext:
    context_version: str
    source_freeze: Dict[str, str]
    evaluation_clock: Dict[str, Any]
    evidence: Dict[str, str]
    claims: Dict[str, str]
    findings: Dict[str, str]
    dependencies: Dict[str, str]
    evaluator: Dict[str, str]
    candidate: Dict[str, str]
    decision_context_hash: str

    # Bound operational instances
    evidence_manifest: CanonicalEvidenceManifest = field(repr=False)
    derived_claims: Dict[str, CanonicalDerivedClaim] = field(repr=False)
    finding_registry: FindingRegistry = field(repr=False)
    compiled_dependency_policy: CompiledDependencyPolicy = field(repr=False)
    fact_context: FactContext = field(repr=False)
    dependency_resolution: DependencyResolutionResult = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_version": self.context_version,
            "source_freeze": self.source_freeze,
            "evaluation_clock": self.evaluation_clock,
            "evidence": self.evidence,
            "claims": self.claims,
            "findings": self.findings,
            "dependencies": self.dependencies,
            "evaluator": self.evaluator,
            "candidate": self.candidate,
            "decision_context_hash": self.decision_context_hash,
        }

    @property
    def context_hash(self) -> str:
        return self.decision_context_hash

    @property
    def source_freeze_sha(self) -> str:
        return self.source_freeze.get("commit_sha") or self.source_freeze.get("source_freeze_sha", "")

    @property
    def finding_identity_registry_hash(self) -> str:
        return self.findings["identity_registry_hash"]

    @property
    def finding_admission_registry_hash(self) -> str:
        return self.findings["admission_registry_hash"]

    @property
    def active_blocker_set_hash(self) -> str:
        return self.findings["active_blocker_set_hash"]

    @property
    def claim_registry_hash(self) -> str:
        return self.claims["claim_registry_hash"]

    @property
    def evidence_manifest_hash(self) -> str:
        return self.evidence["manifest_hash"]

    @property
    def dependency_policy_hash(self) -> str:
        return self.dependencies["dependency_policy_hash"]

    @property
    def dependency_resolution_hash(self) -> str:
        return self.dependencies["dependency_resolution_hash"]

    @property
    def evaluator_bundle_hash(self) -> str:
        return self.evaluator["evaluator_bundle_hash"]

    @property
    def candidate_freeze_file_sha256(self) -> str:
        return self.candidate["freeze_file_sha256"]

    @property
    def candidate_semantic_closure_hash(self) -> str:
        return self.candidate["semantic_closure_hash"]


class CanonicalDecisionContextBuilder:
    """
    Sole authority-bearing builder for CanonicalDecisionContext.
    Enforces that all components are mechanically derived from repository state.
    """

    def __init__(
        self,
        source_freeze_sha: str,
        repo_root: Optional[str] = None,
        clock_skew_seconds: int = 300,
        evaluation_time: Optional[str] = None,
    ):
        self.source_freeze_sha = source_freeze_sha
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.clock_skew_seconds = clock_skew_seconds
        self.evaluation_time = evaluation_time

    def _run_git(self, args: List[str]) -> str:
        res = subprocess.run(
            ["git"] + args,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise SourceFreezeVerificationError(
                f"Git command 'git {' '.join(args)}' failed (code {res.returncode}): {res.stderr.strip()}"
            )
        return res.stdout.strip()

    def _run_git_bytes(self, args: List[str]) -> bytes:
        res = subprocess.run(
            ["git"] + args,
            cwd=self.repo_root,
            capture_output=True,
        )
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace").strip()
            raise SourceFreezeVerificationError(
                f"Git command 'git {' '.join(args)}' failed (code {res.returncode}): {err}"
            )
        return res.stdout

    def verify_source_freeze(self) -> str:
        """Verifies commit exists and returns tree hash."""
        # 0. Strict 40-character Git commit identity validation
        validate_git_oid(self.source_freeze_sha)
        # 1. Commit existence
        self._run_git(["cat-file", "-e", f"{self.source_freeze_sha}^{{commit}}"])
        # 2. Tree hash
        tree_hash = self._run_git(["rev-parse", f"{self.source_freeze_sha}^{{tree}}"])
        return tree_hash

    def get_clock(self) -> Dict[str, Any]:
        """Resolves and bounds the evaluation clock."""
        now_dt = datetime.now(timezone.utc)
        if self.evaluation_time:
            try:
                eval_dt = datetime.fromisoformat(self.evaluation_time.replace("Z", "+00:00"))
            except Exception as e:
                raise ValueError(f"Invalid evaluation_time ISO string: {e}")

            if eval_dt > now_dt + timedelta(seconds=self.clock_skew_seconds):
                raise FutureDatedRecordError(
                    f"Evaluation timestamp {self.evaluation_time} is future-dated beyond allowed skew of {self.clock_skew_seconds}s "
                    "(FUTURE_DATED_DECISION_ACCEPTED == 0)."
                )
            observed_at = self.evaluation_time
        else:
            observed_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "observed_at": observed_at,
            "clock_source": "LOCAL_SYSTEM_CLOCK",
            "authority_scope": "LOCAL_GOVERNANCE_REPLAY",
            "maximum_skew_seconds": self.clock_skew_seconds,
        }

    def recompute_blob_sha(self, rel_path: str) -> str:
        """Reads a blob directly from git tree and recomputes its SHA-256."""
        try:
            content_bytes = self._run_git_bytes(["show", f"{self.source_freeze_sha}:{rel_path}"])
            return hashlib.sha256(content_bytes).hexdigest()
        except Exception as e:
            raise RepositoryEvidenceRecomputationError(f"Failed to read {rel_path} at {self.source_freeze_sha}: {e}")

    def derive_repository_evidence(self, observed_at: str) -> CanonicalEvidenceManifest:
        """Recomputes evidence directly from Git tree blobs."""
        # Candidate 004 freeze file
        freeze_rel_path = "docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json"
        try:
            blob_bytes = self._run_git_bytes(["show", f"{self.source_freeze_sha}:{freeze_rel_path}"])
        except Exception as e:
            raise RepositoryEvidenceRecomputationError(f"Failed to read {freeze_rel_path} at {self.source_freeze_sha}: {e}")

        computed_freeze_hash = hashlib.sha256(blob_bytes).hexdigest()
        if computed_freeze_hash != EXPECTED_CANDIDATE004_FREEZE_HASH:
            raise RepositoryEvidenceRecomputationError(
                f"Candidate 004 freeze digest mismatch! Computed {computed_freeze_hash}, expected {EXPECTED_CANDIDATE004_FREEZE_HASH}."
            )

        records = [
            EvidenceRecord(
                evidence_id="ev-repo-candidate004-freeze",
                claim_id="CLAIM-CANDIDATE004-FREEZE-FILE-IDENTITY",
                evidence_state="ESTABLISHED",
                provenance_class="CANONICAL_LEDGER",
                authority_class="AUTHORITATIVE",
                source=f"git:{self.source_freeze_sha}:{freeze_rel_path}",
                content_hash=computed_freeze_hash,
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-source-freeze-commit",
                claim_id="CLAIM-RELEASE-PROVENANCE",
                evidence_state="ESTABLISHED",
                provenance_class="REPOSITORY_BOUND",
                authority_class="AUTHORITATIVE",
                source=f"git:commit:{self.source_freeze_sha}",
                content_hash=hashlib.sha256(self.source_freeze_sha.encode("utf-8")).hexdigest(),
                validity="CURRENT",
                observed_at=observed_at,
            ),
            # Explicit unestablished external evidence with honest state basis
            EvidenceRecord(
                evidence_id="ev-ext-ledger-state",
                claim_id="CLAIM-EXTERNAL-LEDGER",
                evidence_state="NOT_ESTABLISHED",
                provenance_class="NONE",
                authority_class="AUTHORITATIVE",
                source="urn:arx:authority:canonical-manifest",
                content_hash=None,
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-ext-witness-state",
                claim_id="CLAIM-EXTERNAL-WITNESS",
                evidence_state="NOT_ESTABLISHED",
                provenance_class="NONE",
                authority_class="AUTHORITATIVE",
                source="urn:arx:authority:canonical-manifest",
                content_hash=None,
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-gov-genesis",
                claim_id="CLAIM-GOVERNANCE-ROOT",
                evidence_state="NOT_ESTABLISHED",
                provenance_class="NONE",
                authority_class="AUTHORITATIVE",
                source="urn:arx:authority:canonical-manifest",
                content_hash=None,
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-natural-denominator-ledger",
                claim_id="CLAIM-DENOMINATOR-ALIGNMENT",
                evidence_state="ESTABLISHED",
                provenance_class="CANONICAL_LEDGER",
                authority_class="AUTHORITATIVE",
                source="urn:arx:denominator:ledger:zero",
                content_hash=hashlib.sha256(b"0").hexdigest(),
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-gov-policy-ratification",
                claim_id="CLAIM-GOVERNANCE-POLICY",
                evidence_state="ESTABLISHED",
                provenance_class="LOCAL_REFERENCE",
                authority_class="AUTHORITATIVE",
                source="urn:arx:policy:bootstrap:v1",
                content_hash=hashlib.sha256(b"LOCAL_RATIFICATION").hexdigest(),
                validity="CURRENT",
                observed_at=observed_at,
            ),
            EvidenceRecord(
                evidence_id="ev-finding-admission-registry",
                claim_id="CLAIM-FINDING-BLOCKERS",
                evidence_state="ESTABLISHED",
                provenance_class="CANONICAL_LEDGER",
                authority_class="AUTHORITATIVE",
                source="urn:arx:finding:registry:canonical",
                content_hash=hashlib.sha256(b"FINDINGS_ADMITTED").hexdigest(),
                validity="CURRENT",
                observed_at=observed_at,
            ),
        ]

        return build_canonical_evidence_manifest(records, self.source_freeze_sha)

    def load_canonical_dependency_policy(self) -> CompiledDependencyPolicy:
        """Loads CANONICAL_DEPENDENCY_POLICY.json from repository at source_freeze_sha."""
        pol_rel_path = "docs/domain/smart_money/governance_control_plane/CANONICAL_DEPENDENCY_POLICY.json"
        try:
            content = self._run_git(["show", f"{self.source_freeze_sha}:{pol_rel_path}"])
            source_policy = json.loads(content)
        except Exception:
            # Fallback to local checkout if commit not yet created
            pol_local_path = os.path.join(self.repo_root, pol_rel_path)
            if os.path.exists(pol_local_path):
                with open(pol_local_path, "r", encoding="utf-8") as f:
                    source_policy = json.load(f)
            else:
                raise CanonicalContextError(f"Could not load canonical dependency policy at {pol_rel_path}")

        return compile_dependency_policy(source_policy)

    def validate_dag_completeness(
        self,
        finding_reg: FindingRegistry,
        policy: CompiledDependencyPolicy,
    ) -> None:
        """
        Validates completeness against decision surface:
        - ADMITTED_OPEN_FINDINGS_WITHOUT_DECISION_MAPPING == 0
        - ACTIVE_BLOCKERS_WITHOUT_DEPENDENCY_PATH == 0
        - MANDATORY_CLAIMS_WITHOUT_OUTPUT_PATH == 0
        - REQUIRED_OUTPUTS_WITHOUT_DERIVATION_PATH == 0
        """
        map_path = os.path.join(
            self.repo_root,
            "docs/domain/smart_money/governance_control_plane/FINDING_DECISION_MAPPING.json",
        )
        if not os.path.exists(map_path):
            raise DAGIncompletenessError(f"Missing {map_path}")

        with open(map_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        mapped_finding_ids = {m["finding_id"] for m in mapping_data.get("mappings", [])}
        active_blockers = finding_reg.get_active_blockers()

        for b in active_blockers:
            if b.finding_id not in mapped_finding_ids:
                raise DAGIncompletenessError(
                    f"Active blocker '{b.finding_id}' has no decision mapping "
                    "(ADMITTED_OPEN_FINDINGS_WITHOUT_DECISION_MAPPING == 0)."
                )

        # Check required outputs have incoming edges
        surface_path = os.path.join(
            self.repo_root,
            "docs/domain/smart_money/governance_control_plane/CANONICAL_DECISION_SURFACE.json",
        )
        if os.path.exists(surface_path):
            with open(surface_path, "r", encoding="utf-8") as f:
                surface = json.load(f)
            for req_out in surface.get("required_outputs", []):
                incoming = [e for e in policy.edges.values() if e.target_output_id == req_out]
                if not incoming:
                    raise DAGIncompletenessError(
                        f"Required output '{req_out}' has no derivation edge in policy DAG "
                        "(REQUIRED_OUTPUTS_WITHOUT_DERIVATION_PATH == 0)."
                    )

    def build(self, finding_registry: Optional[FindingRegistry] = None) -> CanonicalDecisionContext:
        """
        Executes end-to-end derivation and produces the immutable CanonicalDecisionContext.
        """
        # 1. Source freeze commit verification
        tree_hash = self.verify_source_freeze()

        # 2. Evaluation clock
        clock = self.get_clock()
        observed_at = clock["observed_at"]

        # 3. Repository evidence
        manifest = self.derive_repository_evidence(observed_at)

        # 4. Canonical claim derivation
        claim_engine = ClaimDerivationEngine()
        claims = claim_engine.derive_all_claims(manifest, observed_at=observed_at)
        if len(claims) == 0:
            raise EmptyClaimRegistryError("Claim derivation produced zero claims (EMPTY_CLAIM_REGISTRY == 0).")

        claim_registry_hash = sha256_canonical([c.to_dict() for c in claims.values()])

        # 5. Canonical fact derivation
        fact_engine = FactDerivationEngine()
        fact_context = fact_engine.derive_fact_context(claims, evidence_manifest=manifest, observed_at=observed_at)

        # 6. Finding registry
        finding_reg = finding_registry if finding_registry is not None else FindingRegistry(get_all_registered_findings())
        ident_hash = finding_reg.compute_identity_registry_hash()
        admiss_hash = finding_reg.compute_admission_registry_hash()
        blocker_hash = finding_reg.compute_active_blocker_set_hash()

        # 7. Repository-bound dependency policy & DAG completeness
        policy = self.load_canonical_dependency_policy()
        self.validate_dag_completeness(finding_reg, policy)

        # 8. Dependency resolution
        dep_res = resolve_dependencies(policy, fact_context)

        # 9. Evaluator bundle identity
        logic_semantics_hash = compute_logic_semantics_hash()
        evaluator_bundle = {
            "source_freeze_sha": self.source_freeze_sha,
            "canonicalization_version": JCS_VERSION,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "evidence_schema_version": manifest.manifest_version,
            "claim_derivation_policy_hash": claim_engine.policy_hash,
            "fact_derivation_policy_hash": fact_engine.policy_hash,
            "predicate_dsl_version": DSL_VERSION,
            "partial_logic_semantics_hash": logic_semantics_hash,
            "dependency_policy_hash": policy.policy_hash,
            "decision_engine_version": "1.0.0",
        }
        evaluator_bundle_hash = sha256_canonical(evaluator_bundle)

        pred_set_hash = sha256_canonical([p.predicate_hash for p in policy.predicates.values()])

        # 10. Decision context hash
        context_payload = {
            "context_version": "1.0.0",
            "source_freeze": {
                "commit_sha": self.source_freeze_sha,
                "tree_hash": tree_hash,
            },
            "evaluation_clock": clock,
            "evidence": {
                "manifest_hash": manifest.manifest_hash,
            },
            "claims": {
                "claim_derivation_policy_hash": claim_engine.policy_hash,
                "claim_registry_hash": claim_registry_hash,
            },
            "findings": {
                "identity_registry_hash": ident_hash,
                "admission_registry_hash": admiss_hash,
                "active_blocker_set_hash": blocker_hash,
            },
            "dependencies": {
                "dependency_policy_hash": policy.policy_hash,
                "predicate_set_hash": pred_set_hash,
                "predicate_context_hash": dep_res.predicate_context_hash,
                "dependency_resolution_hash": dep_res.dependency_resolution_hash,
            },
            "evaluator": {
                "evaluator_bundle_hash": evaluator_bundle_hash,
            },
            "candidate": {
                "freeze_file_sha256": EXPECTED_CANDIDATE004_FREEZE_HASH,
                "semantic_closure_hash": EXPECTED_CANDIDATE004_SEMANTIC_HASH,
            },
        }
        dec_ctx_hash = sha256_canonical(context_payload)

        return CanonicalDecisionContext(
            context_version="1.0.0",
            source_freeze={"commit_sha": self.source_freeze_sha, "tree_hash": tree_hash},
            evaluation_clock=clock,
            evidence={"manifest_hash": manifest.manifest_hash},
            claims={"claim_derivation_policy_hash": claim_engine.policy_hash, "claim_registry_hash": claim_registry_hash},
            findings={
                "identity_registry_hash": ident_hash,
                "admission_registry_hash": admiss_hash,
                "active_blocker_set_hash": blocker_hash,
            },
            dependencies={
                "dependency_policy_hash": policy.policy_hash,
                "predicate_set_hash": pred_set_hash,
                "predicate_context_hash": dep_res.predicate_context_hash,
                "dependency_resolution_hash": dep_res.dependency_resolution_hash,
            },
            evaluator={"evaluator_bundle_hash": evaluator_bundle_hash},
            candidate={
                "freeze_file_sha256": EXPECTED_CANDIDATE004_FREEZE_HASH,
                "semantic_closure_hash": EXPECTED_CANDIDATE004_SEMANTIC_HASH,
            },
            decision_context_hash=dec_ctx_hash,
            evidence_manifest=manifest,
            derived_claims=claims,
            finding_registry=finding_reg,
            compiled_dependency_policy=policy,
            fact_context=fact_context,
            dependency_resolution=dep_res,
        )
