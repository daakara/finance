#!/usr/bin/env python3
"""
HVA-01 Standalone Witness Key Registry & Rollover Attestation Verifier.
Executes independent verification of RFC 8785 canonical hashes and dual-control
signatures for WITNESS_KEY_REGISTRY.json and KEY_ROLLOVER_ATTESTATION.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hva01.witness.registry import WitnessKeyRegistryManager
from hva01.witness.rollover import KeyRolloverManager


def _load_pubkey(val: str) -> bytes:
    """Parse public key from raw 64-character hex or file path."""
    val = val.strip()
    if os.path.isfile(val):
        with open(val, "rb") as f:
            content = f.read().strip()
        try:
            return bytes.fromhex(content.decode("utf-8"))
        except ValueError:
            return content
    try:
        return bytes.fromhex(val)
    except ValueError:
        raise ValueError(f"Unable to parse public key hex: {val}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify HVA-01 Canonical Registry and Rollover Attestations"
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Path to WITNESS_KEY_REGISTRY.json",
    )
    parser.add_argument(
        "--owner-pubkey",
        required=True,
        help="Hex string or file path for trusted Research Owner Ed25519 public key",
    )
    parser.add_argument(
        "--witness-pubkey",
        required=True,
        help="Hex string or file path for trusted Independent Witness Ed25519 public key",
    )
    parser.add_argument(
        "--rollover",
        help="Optional path to KEY_ROLLOVER_ATTESTATION.json",
    )
    parser.add_argument(
        "--custodian-pubkey",
        help="Hex string or file path for trusted Institutional Custodian Ed25519 public key",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("      ARX TERMINAL — WITNESS REGISTRY AUDIT VERIFIER")
    print("=" * 80)

    try:
        owner_pk = _load_pubkey(args.owner_pubkey)
        witness_pk = _load_pubkey(args.witness_pubkey)
    except Exception as e:
        print(f"[!] Public key loading error: {e}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.registry):
        print(f"[!] Registry file not found: {args.registry}", file=sys.stderr)
        return 1

    with open(args.registry, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    print(f"[*] Auditing registry file: {args.registry}")
    reg_res = WitnessKeyRegistryManager.verify_registry(registry_data, owner_pk, witness_pk)

    if not reg_res.is_valid:
        print(f"[!] REGISTRY VERIFICATION FAILED: {reg_res.errors}", file=sys.stderr)
        return 1

    print("[+] REGISTRY VERIFICATION: PASS")
    print(f"    Canonical RFC 8785 Hash: {reg_res.registry_hash}")
    print(f"    Active Primary Key ID:   {reg_res.primary_key_id}")
    print(f"    Escrow Backup Key ID:    {reg_res.backup_key_id}")
    print(f"    Dual Signatures Valid:   {reg_res.signatures_verified} of 2")

    if args.rollover:
        if not args.custodian_pubkey:
            print("[!] Custodian public key required to verify rollover attestation", file=sys.stderr)
            return 1
        custodian_pk = _load_pubkey(args.custodian_pubkey)

        with open(args.rollover, "r", encoding="utf-8") as f:
            roll_data = json.load(f)

        print(f"\n[*] Auditing rollover attestation: {args.rollover}")
        roll_res = KeyRolloverManager.verify_rollover_attestation(
            roll_data, owner_pk, custodian_pk
        )
        if not roll_res.is_valid:
            print(f"[!] ROLLOVER ATTESTATION FAILED: {roll_res.errors}", file=sys.stderr)
            return 1

        print("[+] ROLLOVER ATTESTATION VERIFICATION: PASS")
        print(f"    Attestation ID:      {roll_res.attestation_id}")
        print(f"    Sequence Number:     {roll_res.sequence_number}")
        print(f"    Revoked Key ID:      {roll_res.revoked_key_id}")
        print(f"    Activated Key ID:    {roll_res.activated_key_id}")
        print(f"    Effective Timestamp: {roll_res.effective_timestamp_utc}")
        if roll_res.compromise_suspected_timestamp_utc:
            print(f"    Quarantine Boundary: {roll_res.compromise_suspected_timestamp_utc}")

    print("=" * 80)
    print("      ALL AUDITED RECORDS CONFIRMED VALID & INTACT")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
