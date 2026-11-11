"""
HVA-01 Cryptographic Infrastructure Package.
Provides RFC 8785 canonicalization, RFC 6962-derived Merkle tree commitments,
Ed25519 detached signatures, and low-entropy blinding defenses.
"""

from .canonical import CanonicalJCS, canonical_sha256, canonicalize_json
from .merkle import (
    MerkleTree,
    build_merkle_tree,
    compute_leaf_hash,
    compute_node_hash,
    generate_inclusion_proof,
    verify_inclusion_proof,
)
from .signatures import (
    SyntheticWitnessKey,
    generate_ed25519_keypair,
    get_key_fingerprint,
    sign_detached,
    verify_detached,
)
from .blinding import (
    create_salted_commitment,
    pad_payload,
    unpad_payload,
    verify_salted_commitment,
)

__all__ = [
    "CanonicalJCS",
    "canonicalize_json",
    "canonical_sha256",
    "compute_leaf_hash",
    "compute_node_hash",
    "build_merkle_tree",
    "generate_inclusion_proof",
    "verify_inclusion_proof",
    "MerkleTree",
    "SyntheticWitnessKey",
    "generate_ed25519_keypair",
    "get_key_fingerprint",
    "sign_detached",
    "verify_detached",
    "create_salted_commitment",
    "verify_salted_commitment",
    "pad_payload",
    "unpad_payload",
]
