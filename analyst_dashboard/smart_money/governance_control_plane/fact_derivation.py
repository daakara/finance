"""
Canonical Fact Derivation Subsystem
ARX Terminal — Smart Money

Implements governed fact derivation from canonical claims per CANONICAL_FACT_DERIVATION_POLICY.json.
Enforces that predicate facts strictly inherit authority from claims and prohibits self-asserted authority.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .canonicalization import sha256_canonical
from .claim_derivation import CanonicalDerivedClaim
from .predicate_evaluator import PredicateFact, FactContext


class FactDerivationError(Exception):
    """Base exception for fact derivation errors."""
    pass


class SelfAssertedFactAuthorityError(FactDerivationError):
    """Raised when a fact self-asserts authority without an upstream canonical claim."""
    pass


class FactWithoutClaimBindingError(FactDerivationError):
    """Raised when a fact is constructed without a bound canonical claim."""
    pass


class FactDerivationEngine:
    """
    Executes governed fact derivation from canonical claims.
    """

    def __init__(self, policy_data: Optional[Dict[str, Any]] = None):
        if policy_data is None:
            policy_data = self._load_default_policy()
        self.policy_data = policy_data
        self.policy_id = policy_data.get("policy_id", "urn:arx:policy:fact-derivation:v1")
        self.policy_version = policy_data.get("policy_version", "1.0.0")
        self.rules = policy_data.get("rules", [])
        self.policy_hash = sha256_canonical(self.policy_data)

    @staticmethod
    def _load_default_policy() -> Dict[str, Any]:
        policy_path = os.path.join(
            os.path.dirname(__file__),
            "../../../docs/domain/smart_money/governance_control_plane/CANONICAL_FACT_DERIVATION_POLICY.json",
        )
        if os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "policy_id": "urn:arx:policy:fact-derivation:v1",
            "policy_version": "1.0.0",
            "rules": [],
        }

    def derive_fact_context(
        self,
        claims: Dict[str, CanonicalDerivedClaim],
        evidence_manifest: Optional[Any] = None,
        observed_at: Optional[str] = None,
    ) -> FactContext:
        """
        Derives all predicate facts from canonical claims and returns a verified FactContext.
        """
        if not observed_at:
            observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        facts: List[PredicateFact] = []

        for rule in self.rules:
            fact_id = rule["fact_id"]
            src_claim_id = rule["source_claim_id"]
            val_type = rule.get("output_type", "string")

            if src_claim_id not in claims:
                raise FactWithoutClaimBindingError(
                    f"Fact '{fact_id}' requires claim '{src_claim_id}', but claim was not derived."
                )

            claim = claims[src_claim_id]

            # Fact inherits authority and validity strictly from source claim
            if claim.claim_state in {"ESTABLISHED", "NOT_ESTABLISHED"}:
                auth_class = "AUTHORITATIVE"
                prov_class = "CANONICAL_LEDGER"
                val = claim.claim_state
            elif claim.claim_state == "CONFLICT":
                auth_class = "NON_AUTHORITATIVE"
                prov_class = "CONFLICTING"
                val = "CONFLICT"
            elif claim.claim_state == "STALE":
                auth_class = "NON_AUTHORITATIVE"
                prov_class = "STALE"
                val = "STALE"
            elif claim.claim_state == "REVOKED":
                auth_class = "NON_AUTHORITATIVE"
                prov_class = "REVOKED"
                val = "REVOKED"
            else:  # UNKNOWN
                auth_class = "NON_AUTHORITATIVE"
                prov_class = "NONE"
                val = "UNKNOWN"

            source_ev_id = claim.evidence_refs[0] if claim.evidence_refs else None

            fact = PredicateFact(
                fact_id=fact_id,
                value=val,
                value_type=val_type,
                source_evidence_id=source_ev_id,
                provenance={
                    "authority_class": auth_class,
                    "provenance_class": prov_class,
                    "source_claim_id": src_claim_id,
                    "claim_hash": claim.claim_hash,
                    "derivation_policy_hash": self.policy_hash,
                    "evidence_id": source_ev_id,
                },
                validity="CURRENT" if claim.claim_state != "STALE" else "STALE",
                establishment="ESTABLISHED" if claim.claim_state == "ESTABLISHED" else "NOT_ESTABLISHED",
                observed_at=observed_at,
            )
            facts.append(fact)

        strict_check = True if evidence_manifest is not None else False
        ctx = FactContext(
            facts=facts,
            evidence_manifest=evidence_manifest,
            strict_authoritative_check=strict_check,
        )
        return ctx

