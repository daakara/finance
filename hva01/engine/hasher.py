"""
Canonical Hasher and Tamper-Evident Hash Chain Engine for HVA-01.
Implements RFC 8785 canonical JSON formatting, length-prefixed framing, and SHA-256 chain verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from ..models.response import ResponseRecord


class HashChainVerificationError(ValueError):
    """Raised when recursive hash-chain integrity verification fails."""
    pass


class CanonicalHasher:
    """
    Deterministic cryptographic hasher for session records.
    
    SECURITY NOTE:
    Local hash chaining provides tamper-evidence relative to a trusted head or genesis hash.
    It does not prevent an operator with unrestricted write permissions from rewriting the log
    and recomputing all hashes unless anchored to an independent external witness (Gate C3).
    """

    @staticmethod
    def canonical_json(data: Dict[str, Any]) -> str:
        """
        Encode dictionary to deterministic JSON string adhering to RFC 8785:
        - Keys sorted lexicographically by UTF-16 code units (Python sort_keys=True matches ASCII/UTF-8)
        - No extraneous whitespace (separators=(',', ':'))
        - UTF-8 encoding
        """
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def compute_genesis_hash(
        cls,
        protocol_document_sha256: str,
        package_sha256: str,
        reviewer_pseudonym: str,
        session_id: str,
    ) -> str:
        """
        Compute deterministic genesis hash H_0:
        H_0 = SHA-256("GENESIS-V1:" || ProtocolDoc || ":" || Package || ":" || Pseudonym || ":" || SessionID)
        """
        components = [
            "GENESIS-V1",
            protocol_document_sha256.lower().strip(),
            package_sha256.lower().strip(),
            reviewer_pseudonym.strip(),
            session_id.strip(),
        ]
        framed_payload = ":".join(components).encode("utf-8")
        return hashlib.sha256(framed_payload).hexdigest()

    @classmethod
    def compute_idempotency_key(cls, session_id: str, case_id: str, sequence_index: int) -> str:
        """
        Compute deterministic idempotency key for submission requests:
        IdempotencyKey = SHA-256("IDEMPOTENCY:" || SessionID || ":" || CaseID || ":" || Sequence)
        """
        framed = f"IDEMPOTENCY:{session_id}:{case_id}:{sequence_index}".encode("utf-8")
        return hashlib.sha256(framed).hexdigest()

    @classmethod
    def compute_record_hash(
        cls,
        previous_hash: str,
        sequence_index: int,
        case_id: str,
        timestamp_utc: str,
        ratings_canonical_json: str,
    ) -> str:
        """
        Compute length-prefixed, collision-resistant record hash H_k:
        H_k = SHA-256(H_{k-1} || "|" || Sequence || "|" || CaseID || "|" || Timestamp || "|" || Len(Payload) || "|" || Payload)
        """
        payload_bytes = ratings_canonical_json.encode("utf-8")
        prefix = f"{previous_hash}|{sequence_index}|{case_id}|{timestamp_utc}|{len(payload_bytes)}|".encode("utf-8")
        return hashlib.sha256(prefix + payload_bytes).hexdigest()

    @classmethod
    def verify_chain(
        cls,
        records: List[ResponseRecord],
        genesis_hash: str,
        expected_total_cases: int = 59,
    ) -> Tuple[bool, Optional[str]]:
        """
        Recursively verify the cryptographic integrity of a sequence of response records.
        Returns (True, None) on success or (False, error_message) on first detected failure.
        """
        if not records:
            return True, None

        prev_hash = genesis_hash
        seen_sequences = set()
        seen_case_ids = set()

        for idx, rec in enumerate(records):
            expected_seq = idx + 1

            # Sequence continuity check
            if rec.sequence_index != expected_seq:
                return False, (
                    f"Sequence discontinuity at index {idx}: expected {expected_seq}, "
                    f"got {rec.sequence_index}"
                )
            if rec.sequence_index in seen_sequences:
                return False, f"Duplicate sequence index detected: {rec.sequence_index}"
            seen_sequences.add(rec.sequence_index)

            if rec.case_id in seen_case_ids:
                return False, f"Duplicate case_id in single session: {rec.case_id}"
            seen_case_ids.add(rec.case_id)

            # Previous hash linkage check
            if rec.previous_hash != prev_hash:
                return False, (
                    f"Hash link fault at sequence {rec.sequence_index} ({rec.case_id}): "
                    f"recorded previous_hash {rec.previous_hash} != expected {prev_hash}"
                )

            # Recompute record hash
            ratings_json = cls.canonical_json(rec.ratings.to_canonical_dict())
            computed_hash = cls.compute_record_hash(
                previous_hash=rec.previous_hash,
                sequence_index=rec.sequence_index,
                case_id=rec.case_id,
                timestamp_utc=rec.timestamp_utc,
                ratings_canonical_json=ratings_json,
            )

            if rec.record_hash != computed_hash:
                return False, (
                    f"Record tampering detected at sequence {rec.sequence_index} ({rec.case_id}): "
                    f"recorded record_hash {rec.record_hash} != recomputed {computed_hash}"
                )

            # Check idempotency key matches
            expected_idemp = cls.compute_idempotency_key(rec.session_id, rec.case_id, rec.sequence_index)
            if rec.idempotency_key != expected_idemp:
                return False, (
                    f"Idempotency key mismatch at sequence {rec.sequence_index}: "
                    f"recorded {rec.idempotency_key} != expected {expected_idemp}"
                )

            # Advance previous hash
            prev_hash = rec.record_hash

        return True, None
