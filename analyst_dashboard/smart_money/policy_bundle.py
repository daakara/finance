"""
Immutable Policy Bundle Authority for ARX Terminal Smart Money Wave 2C Certification.
Binds the exact content, versions, and SHA256 hashes of all 11 governing policies.
Ensures that existing receipts can never be silently reinterpreted under future policy versions.
"""

import hashlib
from typing import Dict, Any, List
from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash

POLICIES = {
    "SOURCE_CONTRACT": {
        "name": "SOURCE_CONTRACT",
        "version": "1.0.0",
        "content": (
            "AUTHORITY: US Securities and Exchange Commission (SEC) EDGAR System.\n"
            "ENDPOINT: https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json\n"
            "DOCUMENTS: SEC Form 4 (Statement of Changes in Beneficial Ownership) and Form 4/A (Amendments).\n"
            "IDENTIFIER: 20-character accession number.\n"
            "KNOWLEDGE_TIME: Exact public_availability_at derived from acceptanceDateTime.\n"
            "FALLBACK: No T22 or ingestion-time fallbacks permitted. Naive datetimes fail closed."
        ),
    },
    "DISCOVERY_POLICY": {
        "name": "DISCOVERY_POLICY",
        "version": "1.0.0",
        "content": (
            "SCOPE: 106 governed common-equity partitions in UB_SM_20261010_R1_19ef857fec1f.\n"
            "ENUMERATION: Exhaustive partition-by-partition discovery across required partition set.\n"
            "COMPLETENESS: Processing all observed events does not prove discovery completeness.\n"
            "HTTP success does not prove discovery completeness.\n"
            "Zero new events observed does not prove source closed.\n"
            "Discovery failures are logged as DISCOVERY_GAP and block certification."
        ),
    },
    "PAGINATION_POLICY": {
        "name": "PAGINATION_POLICY",
        "version": "1.0.0",
        "content": (
            "MODEL: filings.recent primary array plus filings.files continuation arrays.\n"
            "EXHAUSTION: Pagination must be fully exhausted (pagination_exhausted = true).\n"
            "CONTINUATION: Zero continuation tokens may remain at certification (continuation_token_remaining = false).\n"
            "INTEGRITY: Loop detection, missing page detection, and truncated response detection fail closed."
        ),
    },
    "OVERLAP_POLICY": {
        "name": "OVERLAP_POLICY",
        "version": "1.0.0",
        "content": (
            "WINDOW_SECONDS: 86400 (24 hours anti-entropy lookback).\n"
            "RE_ENUMERATION: Every cycle re-enumerates prior source time.\n"
            "IDEMPOTENCY: Known accessions are NOOPs.\n"
            "BREACH: Any previously unseen event with public_availability_at <= certified_watermark "
            "is classified as WATERMARK_INTEGRITY_BREACH and freezes advancement."
        ),
    },
    "HOLDBACK_POLICY": {
        "name": "HOLDBACK_POLICY",
        "version": "1.0.0",
        "content": (
            "WINDOW_SECONDS: 86400 (24 hours certification holdback).\n"
            "STABILITY: Candidate frontier T must satisfy governed holdback window before certification.\n"
            "REENUMERATION: Post-holdback re-enumeration must confirm accession set stability through T.\n"
            "NEW_EVENT: Any new event discovered at or before T after holdback marks candidate NOT_CERTIFIABLE."
        ),
    },
    "GAP_POLICY": {
        "name": "GAP_POLICY",
        "version": "1.0.0",
        "content": (
            "CLASSIFICATION: Exhaustive closed set: DISCOVERY_FAILURE, FETCH_FAILURE, PARSE_FAILURE, "
            "RATE_LIMITED, TIMEOUT, UNKNOWN_SOURCE_WINDOW, AMENDMENT_UNRESOLVED, SOURCE_INTEGRITY_CONFLICT.\n"
            "FRONTIER_RULE: Earliest open or retrying gap strictly clamps candidate completeness.\n"
            "Subsequent filings do not clear or skip the gap.\n"
            "LIFECYCLE: Durable append-only event ledger: GAP_OPENED, GAP_RETRY_ATTEMPTED, GAP_CLASSIFIED, "
            "GAP_RESOLVED, GAP_TERMINALLY_EXCLUDED."
        ),
    },
    "WATERMARK_POLICY": {
        "name": "WATERMARK_POLICY",
        "version": "1.0.0",
        "content": (
            "DEFINITION: Global certified completeness watermark equals the minimum certified partition frontier "
            "across all 106 required partitions.\n"
            "MONOTONICITY: Watermark advancement is strictly monotonic.\n"
            "RECEIPT: Every advancement must produce a cryptographically sealed, immutable receipt."
        ),
    },
    "AVAILABILITY_POLICY": {
        "name": "AVAILABILITY_POLICY",
        "version": "1.0.0",
        "content": (
            "PRECEDENCE: EXACT_SEC_ACCEPTANCE > UNKNOWN.\n"
            "TIMEZONE: Must include explicit UTC timezone offset or 'Z'. Naive datetimes rejected.\n"
            "IMMUTABILITY: Public availability timestamp can NEVER be rewritten to an earlier time in the ledger."
        ),
    },
    "FRESHNESS_POLICY": {
        "name": "FRESHNESS_POLICY",
        "version": "1.0.0",
        "content": (
            "DERIVATION: Freshness is strictly derived from certified completeness watermark, "
            "never from wall clock, latest filing, or processing cursor.\n"
            "TIERS: FRESH (<=24h), STALE (24h-72h), UNAVAILABLE (>72h or no watermark).\n"
            "BREACH: Active integrity breach or open gap forces freshness to DATA_INCOMPLETE or UNKNOWN."
        ),
    },
    "STATE_COMPOSITION_POLICY": {
        "name": "STATE_COMPOSITION_POLICY",
        "version": "1.0.0",
        "content": (
            "FORMAL_STATE: (SCOPE, PIPELINE, COMPLETENESS, FRESHNESS, GAP, SCAN_RESULT, DECISION_ELIGIBILITY).\n"
            "SCOPE_KIND: GOVERNED_SUBSET (106 common equities).\n"
            "MARKET_WIDE: MARKET_WIDE_CAPABILITY_STATE is strictly NOT_AUTHORIZED.\n"
            "GLOBAL_BOOLEANS: Global available, fresh, and no_matches booleans are PROHIBITED without scope qualifier."
        ),
    },
    "CANONICALIZATION_CONTRACT": {
        "name": "CANONICALIZATION_CONTRACT",
        "version": "SM_CERT_CANON_V1",
        "content": (
            "SPEC: SM_CERT_CANON_V1.\n"
            "RULES: Unicode NFC normalization, lexicographically sorted object keys, fixed separators (',', ':'), "
            "explicit nulls, strict prohibition of floats in certification payloads, duplicate key rejection."
        ),
    },
}


class PolicyBundleAuthority:
    """
    Manages and cryptographically seals the 11 governing policies.
    """

    @classmethod
    def get_policy(cls, name: str) -> Dict[str, str]:
        if name not in POLICIES:
            raise KeyError(f"UNKNOWN_POLICY: {name}")
        p = POLICIES[name]
        content_hash = hashlib.sha256(p["content"].encode("utf-8")).hexdigest()
        return {
            "name": p["name"],
            "version": p["version"],
            "content": p["content"],
            "sha256": content_hash,
        }

    @classmethod
    def get_all_policies(cls) -> List[Dict[str, str]]:
        return [cls.get_policy(name) for name in sorted(POLICIES.keys())]

    @classmethod
    def compute_policy_bundle_hash(cls) -> str:
        """Computes deterministic SHA256 of canonical policy bundle under SM_CERT_CANON_V1."""
        policies_summary = []
        for p in cls.get_all_policies():
            policies_summary.append({
                "content_sha256": p["sha256"],
                "policy_name": p["name"],
                "policy_version": p["version"],
            })
        return compute_canonical_hash(policies_summary)

    @classmethod
    def export_policy_bundle(cls) -> Dict[str, Any]:
        """Returns the complete policy bundle structure."""
        return {
            "canonicalization_version": "SM_CERT_CANON_V1",
            "policy_bundle_sha256": cls.compute_policy_bundle_hash(),
            "policy_count": len(POLICIES),
            "policies": cls.get_all_policies(),
        }
