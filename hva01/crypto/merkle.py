"""
HVA-01 Merkle Tree Engine.
Implements RFC 6962-derived domain separation (0x00 leaf, 0x01 internal node),
balanced binary tree construction with direct odd-node promotion, and audit proof verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def compute_leaf_hash(canonical_bytes: bytes) -> bytes:
    """
    Compute domain-separated leaf hash: SHA-256(0x00 || canonical_bytes).
    The 0x00 prefix mathematically prevents second-preimage collision with internal nodes.
    """
    return hashlib.sha256(b"\x00" + canonical_bytes).digest()


def compute_node_hash(left_bytes: bytes, right_bytes: bytes) -> bytes:
    """
    Compute domain-separated internal node hash: SHA-256(0x01 || left_32 || right_32).
    Both children must be exactly 32 bytes (SHA-256 output).
    """
    if len(left_bytes) != 32 or len(right_bytes) != 32:
        raise ValueError(
            f"Child hashes must be exactly 32 bytes each. Got {len(left_bytes)} and {len(right_bytes)} bytes."
        )
    return hashlib.sha256(b"\x01" + left_bytes + right_bytes).digest()


def build_merkle_tree(leaf_hashes: List[bytes]) -> Tuple[bytes, List[List[bytes]]]:
    """
    Construct balanced binary Merkle tree with direct odd-node promotion.
    Returns (root_hash_bytes, tree_layers).
    
    If a layer has an odd count 2k+1, the final node is promoted directly to the next layer
    without duplication, preventing duplicate-branch truncation attacks.
    """
    if not leaf_hashes:
        raise ValueError("Cannot construct Merkle tree over an empty leaf list.")

    for i, h in enumerate(leaf_hashes):
        if len(h) != 32:
            raise ValueError(f"Leaf hash at index {i} must be exactly 32 bytes, got {len(h)} bytes.")

    current_layer = list(leaf_hashes)
    layers = [current_layer]

    while len(current_layer) > 1:
        next_layer = []
        i = 0
        n = len(current_layer)
        while i < n:
            if i + 1 < n:
                parent = compute_node_hash(current_layer[i], current_layer[i + 1])
                next_layer.append(parent)
                i += 2
            else:
                # Odd node promoted directly to next layer
                next_layer.append(current_layer[i])
                i += 1
        current_layer = next_layer
        layers.append(current_layer)

    return current_layer[0], layers


def generate_inclusion_proof(
    leaf_index: int,
    tree_layers: List[List[bytes]],
) -> List[Dict[str, str]]:
    """
    Generate the minimal audit path proving that leaf_index belongs to the tree.
    Each proof step contains:
      {"position": "left" | "right", "hash": "sha256:<hex>"}
    """
    if not tree_layers:
        raise ValueError("Tree layers cannot be empty.")

    num_leaves = len(tree_layers[0])
    if leaf_index < 0 or leaf_index >= num_leaves:
        raise IndexError(f"Leaf index {leaf_index} out of range [0, {num_leaves - 1}].")

    proof = []
    idx = leaf_index

    # Traverse layers from bottom (leaves) to root
    for layer in tree_layers[:-1]:
        layer_len = len(layer)
        if idx % 2 == 0:
            # Current node is left child; sibling is right child if present
            if idx + 1 < layer_len:
                sibling_hash = layer[idx + 1]
                proof.append({"position": "right", "hash": f"sha256:{sibling_hash.hex()}"})
            # If idx is odd node at end of layer, it was promoted directly; no sibling added
        else:
            # Current node is right child; sibling is left child
            sibling_hash = layer[idx - 1]
            proof.append({"position": "left", "hash": f"sha256:{sibling_hash.hex()}"})

        # Index in next layer
        idx = idx // 2

    return proof


def verify_inclusion_proof(
    leaf_hash: bytes,
    proof: List[Dict[str, str]],
    expected_root: bytes,
    leaf_index: int,
    tree_size: int,
) -> bool:
    """
    Verify that leaf_hash at leaf_index is included in tree with expected_root and tree_size.
    Fails closed if any hash or index is invalid.
    """
    if len(leaf_hash) != 32 or len(expected_root) != 32:
        return False
    if leaf_index < 0 or leaf_index >= tree_size:
        return False
    if tree_size <= 0:
        return False

    current = leaf_hash
    proof_idx = 0

    idx = leaf_index
    current_layer_size = tree_size

    while current_layer_size > 1:
        is_even = (idx % 2 == 0)
        has_sibling = not (is_even and idx == current_layer_size - 1)

        if has_sibling:
            if proof_idx >= len(proof):
                return False  # Missing proof step
            step = proof[proof_idx]
            proof_idx += 1

            pos = step.get("position")
            sibling_hex = step.get("hash", "")
            if not sibling_hex.startswith("sha256:"):
                return False
            try:
                sibling_bytes = bytes.fromhex(sibling_hex.split(":", 1)[1])
            except ValueError:
                return False

            if len(sibling_bytes) != 32:
                return False

            if pos == "right":
                if not is_even:
                    return False  # Position mismatch with tree geometry
                current = compute_node_hash(current, sibling_bytes)
            elif pos == "left":
                if is_even:
                    return False  # Position mismatch with tree geometry
                current = compute_node_hash(sibling_bytes, current)
            else:
                return False
        else:
            # Node was promoted directly; no sibling step
            pass

        idx = idx // 2
        current_layer_size = (current_layer_size + 1) // 2

    if proof_idx != len(proof):
        return False  # Superfluous proof steps

    return current == expected_root


@dataclass(frozen=True)
class MerkleTree:
    """
    Immutable container representing a completed Merkle tree commitment.
    """
    root_hash: bytes
    tree_size: int
    layers: List[List[bytes]]

    @classmethod
    def from_leaves(cls, leaf_hashes: List[bytes]) -> MerkleTree:
        root, layers = build_merkle_tree(leaf_hashes)
        return cls(root_hash=root, tree_size=len(leaf_hashes), layers=layers)

    @property
    def root_hex(self) -> str:
        return f"sha256:{self.root_hash.hex()}"

    def get_proof(self, leaf_index: int) -> List[Dict[str, str]]:
        return generate_inclusion_proof(leaf_index, self.layers)
