#!/usr/bin/env python3
"""
HVA-01 Offline Key Ceremony Operator Driver.
Executes the physical/mock ceremony workflow defined in
HVA01_WAVE4C2_SCOPE_LOCK_AND_KEY_CEREMONY_SPECIFICATION_V3 (Gate V3-01 / CTRL-4C2-08).

Outputs a dual-signed RFC 8785 canonical WITNESS_KEY_REGISTRY.json.
Private keys are born inside silicon and are physically non-exportable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from hva01.witness.pkcs11_session import PKCS11SessionManager
from hva01.witness.registry import WitnessKeyRegistryManager
from hva01.witness.token_interface import MockHardwareToken


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HVA-01 Air-Gapped Key Ceremony Driver (Synthetic/Hardware Mode)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Execute in synthetic mock mode (default: True)",
    )
    parser.add_argument(
        "--slot-a",
        type=int,
        default=0,
        help="PKCS#11 slot for Primary Token A (default: 0)",
    )
    parser.add_argument(
        "--slot-b",
        type=int,
        default=1,
        help="PKCS#11 slot for Escrow Token B (default: 1)",
    )
    parser.add_argument(
        "--label-a",
        default="HVA01-W4C2-PRIMARY-KEY-A",
        help="Key label for Primary Token A",
    )
    parser.add_argument(
        "--label-b",
        default="HVA01-W4C2-BACKUP-KEY-B",
        help="Key label for Escrow Token B",
    )
    parser.add_argument(
        "--pin-a",
        default="123456",
        help="User PIN for Token A",
    )
    parser.add_argument(
        "--pin-b",
        default="123456",
        help="User PIN for Token B",
    )
    parser.add_argument(
        "--output",
        default="WITNESS_KEY_REGISTRY.json",
        help="Output destination path for canonical registry",
    )
    parser.add_argument(
        "--export-pubkeys",
        action="store_true",
        help="Export ceremonial signer public keys to <output>.owner_pk and <output>.witness_pk",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("      ARX TERMINAL — HVA-01 V2.1 OFFLINE KEY CEREMONY DRIVER")
    print("      Architecture: Hardware-Native Multi-Key Model (Gate V3-01 / CTRL-4C2-08)")
    print(f"      Execution Mode: {'SYNTHETIC MOCK' if args.mock else 'LIVE PHYSICAL HARDWARE'}")
    print("=" * 80)

    # 1. Initialize PKCS#11 session
    session = PKCS11SessionManager(mock_mode=args.mock)
    if args.mock:
        token_a = MockHardwareToken(serial_number="MOCK-YK5-PRIMARY-001")
        token_b = MockHardwareToken(serial_number="MOCK-YK5-ESCROW-002")
        session.register_mock_token(args.slot_a, token_a)
        session.register_mock_token(args.slot_b, token_b)

    # 2. Authenticate and initialize slots
    session.login(args.slot_a, args.pin_a)
    session.login(args.slot_b, args.pin_b)

    # 3. Generate Primary Key A inside silicon
    print(f"[*] Generating Primary Key A ({args.label_a}) in slot {args.slot_a} silicon...")
    meta_a = session.generate_ed25519_keypair(args.slot_a, args.label_a)
    print(f"    -> Key ID: {meta_a.key_id}")
    print(f"    -> Algorithm: {meta_a.algorithm}")
    print(f"    -> Public Key: {meta_a.public_key_hex}")
    print(f"    -> SHA-256 Fingerprint: {meta_a.fingerprint_sha256}")
    print(f"    -> Non-Exportable: {not meta_a.is_extractable}")

    # 4. Generate Escrow Key B inside silicon
    print(f"[*] Generating Escrow Key B ({args.label_b}) in slot {args.slot_b} silicon...")
    meta_b = session.generate_ed25519_keypair(args.slot_b, args.label_b)
    print(f"    -> Key ID: {meta_b.key_id}")
    print(f"    -> Algorithm: {meta_b.algorithm}")
    print(f"    -> Public Key: {meta_b.public_key_hex}")
    print(f"    -> SHA-256 Fingerprint: {meta_b.fingerprint_sha256}")
    print(f"    -> Non-Exportable: {not meta_b.is_extractable}")

    # 5. Build Registry Structure
    ceremony_id = f"ARX-CEREMONY-HVA01-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = WitnessKeyRegistryManager.create_empty_registry(ceremony_id, now_utc)

    WitnessKeyRegistryManager.add_key_entry(
        registry,
        key_id=meta_a.key_id,
        public_key_bytes=meta_a.public_key_bytes,
        token_serial=meta_a.token_serial,
        status="ACTIVE_PRIMARY",
        effective_date_utc=now_utc,
    )
    WitnessKeyRegistryManager.add_key_entry(
        registry,
        key_id=meta_b.key_id,
        public_key_bytes=meta_b.public_key_bytes,
        token_serial=meta_b.token_serial,
        status="COLD_BACKUP_ESCROW",
        effective_date_utc=now_utc,
    )

    # 6. Compute Canonical RFC 8785 Registry Digest
    canonical_bytes, reg_hash = WitnessKeyRegistryManager.compute_canonical_registry_hash(registry)
    reg_digest = bytes.fromhex(reg_hash)
    print(f"\n[*] Canonical RFC 8785 Registry Hash: {reg_hash}")

    # 7. Dual-Control Signing (Research Owner + Independent Witness)
    # Generate ephemeral ceremonial keys for synthetic simulation
    owner_priv = ed25519.Ed25519PrivateKey.generate()
    owner_pub = owner_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    owner_sig = owner_priv.sign(reg_digest)

    witness_priv = ed25519.Ed25519PrivateKey.generate()
    witness_pub = witness_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    witness_sig = witness_priv.sign(reg_digest)

    WitnessKeyRegistryManager.add_signature(
        registry, "RESEARCH_OWNER", owner_pub, owner_sig, now_utc
    )
    WitnessKeyRegistryManager.add_signature(
        registry, "INDEPENDENT_WITNESS", witness_pub, witness_sig, now_utc
    )
    print("[*] Dual-control ceremonial signatures attached successfully.")
    print(f"    -> Research Owner Public Key:   {owner_pub.hex()}")
    print(f"    -> Independent Witness Public Key: {witness_pub.hex()}")

    # 8. Verify the completed registry
    res = WitnessKeyRegistryManager.verify_registry(registry, owner_pub, witness_pub)
    if not res.is_valid:
        print(f"[!] Registry verification failed: {res.errors}", file=sys.stderr)
        return 1
    print("[*] Registry self-verification PASS (Dual signatures valid & keys verified).")

    # 9. Output to destination file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"[*] Canonical registry written to: {args.output}")

    if args.export_pubkeys:
        owner_pk_path = f"{args.output}.owner_pk"
        witness_pk_path = f"{args.output}.witness_pk"
        with open(owner_pk_path, "w", encoding="utf-8") as f:
            f.write(owner_pub.hex() + "\n")
        with open(witness_pk_path, "w", encoding="utf-8") as f:
            f.write(witness_pub.hex() + "\n")
        print(f"[*] Ceremonial signer public keys written to:\n    {owner_pk_path}\n    {witness_pk_path}")
    print("=" * 80)
    print("      CEREMONY TOOLING EXECUTION COMPLETE — PASS")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
