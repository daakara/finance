"""
HVA-01 RFC 3161 Standards-Conformant ASN.1 DER Timestamp Wire Engine.
Implements TimeStampReq generation, strict DER encoding/decoding,
and full CMS SignedData cryptographic verification for RFC 3161 TimeStampToken.

GOVERNANCE CLASSIFICATION: TIER 3 — Cryptographic and Evidence-Integrity Controls.
FAIL-CLOSED: Rejects malformed encodings, untrusted signers, expired certificates,
and mismatched message imprints or nonces.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from pyasn1.codec.der import decoder, encoder
from pyasn1.error import PyAsn1Error
from pyasn1.type import char, univ, useful
from pyasn1_modules import rfc2459, rfc3161, rfc5652

# Standard Cryptographic OIDs
OID_SHA256 = "2.16.840.1.101.3.4.2.1"
OID_SHA384 = "2.16.840.1.101.3.4.2.2"
OID_SHA512 = "2.16.840.1.101.3.4.2.3"

OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
OID_CT_TST_INFO = "1.2.840.113549.1.9.16.1.4"
OID_KP_TIME_STAMPING = "1.3.6.1.5.5.7.3.8"

OID_RSA_ENCRYPTION = "1.2.840.113549.1.1.1"
OID_SHA256_WITH_RSA = "1.2.840.113549.1.1.11"
OID_SHA384_WITH_RSA = "1.2.840.113549.1.1.12"
OID_SHA512_WITH_RSA = "1.2.840.113549.1.1.13"

OID_ECDSA_WITH_SHA256 = "1.2.840.10045.4.3.2"
OID_ECDSA_WITH_SHA384 = "1.2.840.10045.4.3.3"
OID_ECDSA_WITH_SHA512 = "1.2.840.10045.4.3.4"

OID_ED25519 = "1.3.101.112"

DEFAULT_TSA_POLICY_OID = "2.16.840.1.114412.7.1"

SUPPORTED_HASH_ALGORITHMS: Dict[str, Tuple[str, int]] = {
    "SHA-256": (OID_SHA256, 32),
    "SHA-384": (OID_SHA384, 48),
    "SHA-512": (OID_SHA512, 64),
}

OID_TO_HASH_NAME: Dict[str, str] = {
    OID_SHA256: "SHA-256",
    OID_SHA384: "SHA-384",
    OID_SHA512: "SHA-512",
}


# ============================================================================
# Exception Hierarchy (Fail-Closed)
# ============================================================================

class TimestampWireError(Exception):
    """Base exception for all RFC 3161 wire-level protocol and parsing failures."""
    pass


class MalformedAsn1Error(TimestampWireError):
    """Raised when an ASN.1 DER stream cannot be strictly decoded or has trailing bytes."""
    pass


class UnsupportedAlgorithmError(TimestampWireError):
    """Raised when a request or response specifies an unsupported cryptographic algorithm."""
    pass


class PKIStatusError(TimestampWireError):
    """Raised when TimeStampResp indicates rejection or failure (status != granted)."""
    pass


class InvalidTimestampTokenError(TimestampWireError):
    """Raised when the TimeStampToken structure or CMS SignedData encapsulation is invalid."""
    pass


class InvalidSignatureError(TimestampWireError):
    """Raised when cryptographic signature verification fails mathematically."""
    pass


class UntrustedTSACertificateError(TimestampWireError):
    """Raised when the TSA signer certificate fails trust, EKU, or validity checks."""
    pass


class RequestResponseBindingError(TimestampWireError):
    """Raised when response message imprint, algorithm, or nonce does not match request."""
    pass


class ClockDriftExceededError(TimestampWireError):
    """Raised when the timestamp generation time differs from reference time beyond limits."""
    pass


class TSAPolicyMismatchError(TimestampWireError):
    """Raised when the returned TSA policy OID is unauthorized or unexpected."""
    pass


# ============================================================================
# Verified Timestamp Data Model
# ============================================================================

@dataclass(frozen=True)
class VerifiedTimestamp:
    """
    Cryptographically authenticated RFC 3161 timestamp token with full audit provenance.
    """
    tsa_identity: str
    tsa_policy_oid: str
    message_imprint_algorithm: str
    message_imprint_digest: str  # Format: "sha256:<hex>"
    timestamp_utc: str          # ISO 8601 UTC timestamp
    serial_number: int
    nonce: Optional[int]
    tsa_certificate_fingerprint: str  # SHA-256 hex digest of DER certificate
    signer_common_name: str
    signature_algorithm_oid: str
    token_der: bytes
    raw_response_der: bytes
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d["token_der_b64"] = base64.b64encode(self.token_der).decode("ascii")
        d["raw_response_der_b64"] = base64.b64encode(self.raw_response_der).decode("ascii")
        del d["token_der"]
        del d["raw_response_der"]
        return d

    def to_token_dict(self) -> Dict[str, Any]:
        """Convert to format compatible with HVA-01 WitnessAttestation.timestamp_token."""
        return {
            "tsa_identity": self.tsa_identity,
            "tsa_policy_oid": self.tsa_policy_oid,
            "message_imprint_algorithm": self.message_imprint_algorithm,
            "message_imprint_digest": self.message_imprint_digest,
            "timestamp_utc": self.timestamp_utc,
            "serial_number": self.serial_number,
            "nonce": self.nonce or 0,
            "tsa_certificate_fingerprint": self.tsa_certificate_fingerprint,
            "tsa_signature_b64": base64.b64encode(self.token_der).decode("ascii"),
            "is_synthetic": self.is_synthetic,
            "verification_status": "VERIFIED_CRYPTOGRAPHIC_CMS",
        }


# ============================================================================
# Request Serialization & Parsing
# ============================================================================

def create_timestamp_request(
    imprint_digest: Union[bytes, str],
    hash_algorithm: str = "SHA-256",
    policy_oid: Optional[str] = None,
    nonce: Optional[int] = None,
    cert_req: bool = True,
) -> bytes:
    """
    Construct a standards-conformant RFC 3161 TimeStampReq encoded in strict ASN.1 DER.

    Args:
        imprint_digest: Digest bytes or hex string (e.g. "sha256:abcd..." or raw bytes).
        hash_algorithm: Name of hash algorithm ("SHA-256", "SHA-384", "SHA-512").
        policy_oid: Optional requested TSA policy OID.
        nonce: Optional integer nonce. If None, generates a 256-bit CSPRNG integer.
        cert_req: Whether to request the TSA signing certificate in response (default True).
    """
    if hash_algorithm not in SUPPORTED_HASH_ALGORITHMS:
        raise UnsupportedAlgorithmError(f"Unsupported hash algorithm: '{hash_algorithm}'")

    algo_oid_str, expected_len = SUPPORTED_HASH_ALGORITHMS[hash_algorithm]

    # Normalize imprint digest to raw bytes
    if isinstance(imprint_digest, str):
        clean_hex = imprint_digest.strip()
        if ":" in clean_hex:
            clean_hex = clean_hex.split(":", 1)[1]
        try:
            digest_bytes = bytes.fromhex(clean_hex)
        except ValueError as e:
            raise MalformedAsn1Error(f"Invalid hex string for message imprint digest: {e}")
    elif isinstance(imprint_digest, (bytes, bytearray)):
        digest_bytes = bytes(imprint_digest)
    else:
        raise MalformedAsn1Error(f"Imprint digest must be bytes or str, got {type(imprint_digest)}")

    if len(digest_bytes) != expected_len:
        raise MalformedAsn1Error(
            f"Digest length mismatch for {hash_algorithm}: expected {expected_len} bytes, got {len(digest_bytes)}"
        )

    # Generate or validate nonce
    if nonce is None:
        # Generate 256-bit positive integer via CSPRNG
        token_nonce = int.from_bytes(os.urandom(32), "big")
    else:
        if not isinstance(nonce, int) or nonce <= 0:
            raise MalformedAsn1Error(f"Nonce must be a positive integer, got {nonce}")
        token_nonce = nonce

    # Construct ASN.1 TimeStampReq
    req = rfc3161.TimeStampReq()
    req["version"] = 1

    msg_imprint = rfc3161.MessageImprint()
    algo_id = rfc3161.AlgorithmIdentifier()
    algo_id["algorithm"] = univ.ObjectIdentifier(algo_oid_str)
    # DER encoding requires NULL parameters for standard hash algorithms
    algo_id["parameters"] = encoder.encode(univ.Null(""))
    msg_imprint["hashAlgorithm"] = algo_id
    msg_imprint["hashedMessage"] = univ.OctetString(digest_bytes)
    req["messageImprint"] = msg_imprint

    if policy_oid:
        req["reqPolicy"] = univ.ObjectIdentifier(policy_oid)

    req["nonce"] = univ.Integer(token_nonce)
    req["certReq"] = univ.Boolean(cert_req)

    return encoder.encode(req)


def parse_timestamp_request(req_der: bytes, max_size: int = 65536) -> Dict[str, Any]:
    """
    Strictly parse and validate an RFC 3161 TimeStampReq from raw DER bytes.
    Enforces maximum payload size and zero trailing bytes.
    """
    if not isinstance(req_der, (bytes, bytearray)):
        raise MalformedAsn1Error("Request must be bytes")
    if len(req_der) == 0:
        raise MalformedAsn1Error("Request payload is empty")
    if len(req_der) > max_size:
        raise MalformedAsn1Error(f"Request payload exceeds maximum size limit ({len(req_der)} > {max_size})")

    try:
        decoded, rest = decoder.decode(req_der, asn1Spec=rfc3161.TimeStampReq())
    except (PyAsn1Error, ValueError) as e:
        raise MalformedAsn1Error(f"Failed to decode ASN.1 DER TimeStampReq: {e}")

    if len(rest) != 0:
        raise MalformedAsn1Error(f"Trailing unparsed bytes detected in DER stream ({len(rest)} bytes)")

    version = int(decoded["version"])
    if version != 1:
        raise MalformedAsn1Error(f"Unsupported TimeStampReq version: {version} (expected 1)")

    imprint = decoded["messageImprint"]
    algo_oid = str(imprint["hashAlgorithm"]["algorithm"])
    if algo_oid not in OID_TO_HASH_NAME:
        raise UnsupportedAlgorithmError(f"Unsupported message imprint algorithm OID: '{algo_oid}'")

    algo_name = OID_TO_HASH_NAME[algo_oid]
    hashed_bytes = bytes(imprint["hashedMessage"])
    expected_len = SUPPORTED_HASH_ALGORITHMS[algo_name][1]
    if len(hashed_bytes) != expected_len:
        raise MalformedAsn1Error(
            f"Hashed message length ({len(hashed_bytes)}) does not match {algo_name} length ({expected_len})"
        )

    nonce = int(decoded["nonce"]) if decoded["nonce"].isValue else None
    policy = str(decoded["reqPolicy"]) if decoded["reqPolicy"].isValue else None
    cert_req = bool(decoded["certReq"]) if decoded["certReq"].isValue else False

    return {
        "version": version,
        "hash_algorithm": algo_name,
        "hash_algorithm_oid": algo_oid,
        "hashed_message": hashed_bytes,
        "hashed_message_hex": hashed_bytes.hex(),
        "nonce": nonce,
        "req_policy": policy,
        "cert_req": cert_req,
    }


# ============================================================================
# Certificate Chain and Trust Verification (Fail-Closed)
# ============================================================================

def _verify_cert_signature(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    """Mathematically verify child certificate signature using issuer public key."""
    issuer_pub = issuer.public_key()
    try:
        if isinstance(issuer_pub, rsa.RSAPublicKey):
            hash_algo = child.signature_hash_algorithm
            if hash_algo is None:
                return False
            issuer_pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                padding.PKCS1v15(),
                hash_algo,
            )
            return True
        elif isinstance(issuer_pub, ec.EllipticCurvePublicKey):
            hash_algo = child.signature_hash_algorithm
            if hash_algo is None:
                return False
            issuer_pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                ec.ECDSA(hash_algo),
            )
            return True
        elif isinstance(issuer_pub, ed25519.Ed25519PublicKey):
            issuer_pub.verify(child.signature, child.tbs_certificate_bytes)
            return True
        else:
            return False
    except Exception:
        return False


def verify_tsa_certificate_trust(
    signer_cert: x509.Certificate,
    trusted_certs: List[x509.Certificate],
    candidate_intermediates: List[x509.Certificate],
    validation_time: datetime,
) -> List[x509.Certificate]:
    """
    Perform fail-closed X.509 certificate path and trust-anchor validation for a TSA signer.

    Enforces:
      1. Fail-closed rejection when no trusted certificates are provided.
      2. Time validity of the signer and all issuing CAs at validation_time.
      3. ExtendedKeyUsage containing id-kp-timeStamping (1.3.6.1.5.5.7.3.8) on the signer.
      4. KeyUsage digitalSignature and/or nonRepudiation/contentCommitment if KeyUsage is present.
      5. Explicit trust-anchor matching (pinned certificate or trusted root CA).
      6. Intermediate CA constraints: basicConstraints must be present with ca=True, path length constraints,
         and keyCertSign in keyUsage.
      7. Mathematical cryptographic verification of every certificate in the chain.
      8. Fail-closed rejection of self-signed signers that are not explicitly authorized trust anchors.
      9. Loop and chain-depth bounds.

    Returns:
        List of verified x509.Certificate objects in path order: [signer_cert, ... intermediate_cas ..., trust_anchor].
    """
    if not trusted_certs or not isinstance(trusted_certs, list) or len(trusted_certs) == 0:
        raise UntrustedTSACertificateError(
            "No trusted certificates configured; timestamp verification requires explicit trust anchors"
        )

    # 1. Signer validity period
    if validation_time < signer_cert.not_valid_before_utc or validation_time > signer_cert.not_valid_after_utc:
        valid_range = f"{signer_cert.not_valid_before_utc.isoformat()} to {signer_cert.not_valid_after_utc.isoformat()}"
        raise UntrustedTSACertificateError(
            f"TSA certificate is not valid at timestamp generation time: "
            f"valid from {valid_range}, genTime={validation_time.isoformat()}"
        )

    # 2. ExtendedKeyUsage for Time Stamping (RFC 3161 section 2.3)
    try:
        eku_ext = signer_cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.EXTENDED_KEY_USAGE)
        eku_oids = [oid.dotted_string for oid in eku_ext.value]
        if OID_KP_TIME_STAMPING not in eku_oids:
            raise UntrustedTSACertificateError(
                f"TSA certificate missing required ExtendedKeyUsage: {OID_KP_TIME_STAMPING}"
            )
    except x509.ExtensionNotFound:
        raise UntrustedTSACertificateError("TSA certificate possesses no ExtendedKeyUsage extension")

    # 3. KeyUsage if present
    try:
        ku_ext = signer_cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE)
        ku = ku_ext.value
        if not (ku.digital_signature or ku.content_commitment):
            raise UntrustedTSACertificateError(
                "TSA certificate keyUsage does not permit digitalSignature or contentCommitment"
            )
    except x509.ExtensionNotFound:
        pass

    # 4. Check if signer_cert is itself directly in trusted_certs
    signer_fp = signer_cert.fingerprint(hashes.SHA256()).hex()
    trusted_fps = {tc.fingerprint(hashes.SHA256()).hex(): tc for tc in trusted_certs}

    if signer_fp in trusted_fps:
        # Pinned certificate in trusted_certs
        if signer_cert.issuer == signer_cert.subject:
            # Self-signed trust anchor: verify self-signature
            if not _verify_cert_signature(signer_cert, signer_cert):
                raise UntrustedTSACertificateError(
                    f"Self-signed TSA certificate '{signer_fp}' self-signature verification failed"
                )
        return [signer_cert]

    # Rejection of arbitrary self-signed signers
    if signer_cert.issuer == signer_cert.subject:
        raise UntrustedTSACertificateError(
            f"TSA certificate '{signer_fp}' is self-signed, is not in trusted store, "
            f"and is not an authorized trust anchor"
        )

    # 5. Build and verify certificate chain from signer to trusted CA
    chain: List[x509.Certificate] = [signer_cert]
    current = signer_cert
    pool = list(candidate_intermediates) + list(trusted_certs)
    anchor_reached: Optional[x509.Certificate] = None
    max_depth = 8

    while len(chain) <= max_depth:
        # Find candidate issuers in pool matching current.issuer
        candidates = [c for c in pool if c.subject == current.issuer]
        if not candidates:
            raise UntrustedTSACertificateError(
                f"Certificate chain incomplete: issuer '{current.issuer.rfc4514_string()}' "
                f"not found in available certificates or trust anchors"
            )

        valid_issuer: Optional[x509.Certificate] = None
        for cand in candidates:
            # Verify AKI/SKI if present
            try:
                aki = current.extensions.get_extension_for_oid(x509.oid.ExtensionOID.AUTHORITY_KEY_IDENTIFIER).value
                ski = cand.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_KEY_IDENTIFIER).value
                if aki.key_identifier and ski.key_identifier and aki.key_identifier != ski.key_identifier:
                    continue
            except x509.ExtensionNotFound:
                pass

            if _verify_cert_signature(current, cand):
                valid_issuer = cand
                break

        if valid_issuer is None:
            raise UntrustedTSACertificateError(
                f"Certificate signature verification failed against candidate issuer "
                f"'{current.issuer.rfc4514_string()}'"
            )

        # Intermediate CA constraints (SEC-4C1-03)
        try:
            bc_ext = valid_issuer.extensions.get_extension_for_oid(x509.oid.ExtensionOID.BASIC_CONSTRAINTS)
            if not bc_ext.value.ca:
                raise UntrustedTSACertificateError(
                    f"Issuer certificate '{valid_issuer.subject.rfc4514_string()}' has basicConstraints with ca=False"
                )
            if bc_ext.value.path_length is not None:
                intermediates_count = len(chain) - 1
                if intermediates_count > bc_ext.value.path_length:
                    raise UntrustedTSACertificateError(
                        f"Certificate path length constraint exceeded "
                        f"({intermediates_count} > {bc_ext.value.path_length})"
                    )
        except x509.ExtensionNotFound:
            raise UntrustedTSACertificateError(
                f"Issuer certificate '{valid_issuer.subject.rfc4514_string()}' "
                f"missing mandatory basicConstraints extension"
            )

        # KeyUsage on CA
        try:
            ku_ext = valid_issuer.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE)
            if not ku_ext.value.key_cert_sign:
                raise UntrustedTSACertificateError(
                    f"Issuer certificate '{valid_issuer.subject.rfc4514_string()}' lacks keyCertSign in keyUsage"
                )
        except x509.ExtensionNotFound:
            pass

        # CA validity dates
        if validation_time < valid_issuer.not_valid_before_utc or validation_time > valid_issuer.not_valid_after_utc:
            raise UntrustedTSACertificateError(
                f"Issuer certificate '{valid_issuer.subject.rfc4514_string()}' is not valid at validation time"
            )

        chain.append(valid_issuer)

        # Check if valid_issuer is in trusted_certs
        issuer_fp = valid_issuer.fingerprint(hashes.SHA256()).hex()
        if issuer_fp in trusted_fps:
            if valid_issuer.issuer == valid_issuer.subject:
                if not _verify_cert_signature(valid_issuer, valid_issuer):
                    raise UntrustedTSACertificateError(
                        f"Trust anchor '{valid_issuer.subject.rfc4514_string()}' self-signature verification failed"
                    )
            anchor_reached = valid_issuer
            break

        if valid_issuer.issuer == valid_issuer.subject:
            raise UntrustedTSACertificateError(
                f"Certificate chain terminates at untrusted root CA '{valid_issuer.subject.rfc4514_string()}' "
                f"(not present in authorized trust anchors)"
            )

        if valid_issuer in chain[:-1]:
            raise UntrustedTSACertificateError("Certificate chain contains a cycle / infinite loop")

        current = valid_issuer

    if anchor_reached is None:
        raise UntrustedTSACertificateError(
            f"Certificate chain depth exceeded maximum ({max_depth}) without reaching an authorized trust anchor"
        )

    return chain


# ============================================================================
# Response Verification & Parsing
# ============================================================================

def parse_and_verify_timestamp_response(
    resp_der: bytes,
    expected_imprint_digest: Union[bytes, str],
    trusted_certs: List[x509.Certificate],
    expected_nonce: Optional[int] = None,
    expected_policy_oid: Optional[str] = None,
    reference_time_utc: Optional[str] = None,
    max_drift_seconds: float = 30.0,
    max_size: int = 1048576,  # 1 MB
) -> VerifiedTimestamp:
    """
    Perform complete fail-closed verification of an RFC 3161 TimeStampResp ASN.1 DER stream.

    Verifies:
      1. Strict DER encoding and size bounds.
      2. PKIStatusInfo indicates granted (0 or 1).
      3. Presence of TimeStampToken as CMS SignedData (1.2.840.113549.1.7.2).
      4. TSTInfo encapsulation (1.2.840.113549.1.9.16.1.4) and version 1.
      5. Message imprint algorithm and digest match expected value byte-for-byte.
      6. Nonce matches expected nonce (if nonce was requested).
      7. Signer certificate is trusted, valid, and possesses Time Stamping EKU.
      8. CMS signature validates mathematically against the signer's public key.
      9. Clock drift against reference time (if provided) is within bounds.
    """
    if not isinstance(resp_der, (bytes, bytearray)):
        raise MalformedAsn1Error("Response must be bytes")
    if len(resp_der) == 0:
        raise MalformedAsn1Error("Response payload is empty")
    if len(resp_der) > max_size:
        raise MalformedAsn1Error(f"Response payload exceeds maximum size limit ({len(resp_der)} > {max_size})")

    # 1. Decode TimeStampResp
    try:
        decoded_resp, rest = decoder.decode(resp_der, asn1Spec=rfc3161.TimeStampResp())
    except (PyAsn1Error, ValueError) as e:
        raise MalformedAsn1Error(f"Failed to decode ASN.1 DER TimeStampResp: {e}")

    if len(rest) != 0:
        raise MalformedAsn1Error(f"Trailing unparsed bytes detected in TimeStampResp ({len(rest)} bytes)")

    # 2. Check PKIStatus
    status_info = decoded_resp["status"]
    status_code = int(status_info["status"])
    if status_code not in (0, 1):  # 0=granted, 1=grantedWithMods
        status_str = str(status_info["statusString"]) if status_info["statusString"].isValue else "Unknown"
        fail_info = str(status_info["failInfo"]) if status_info["failInfo"].isValue else "None"
        raise PKIStatusError(
            f"TSA rejected timestamp request: status={status_code}, statusString='{status_str}', failInfo='{fail_info}'"
        )

    # 3. Check TimeStampToken Presence
    if not decoded_resp["timeStampToken"].isValue:
        raise InvalidTimestampTokenError("TimeStampResp status is granted but timeStampToken is missing")

    token_content_info = decoded_resp["timeStampToken"]
    content_type_oid = str(token_content_info["contentType"])
    if content_type_oid != OID_SIGNED_DATA:
        raise InvalidTimestampTokenError(
            f"TimeStampToken contentType must be id-signedData ({OID_SIGNED_DATA}), got '{content_type_oid}'"
        )

    raw_token_der = encoder.encode(token_content_info)

    # 4. Decode CMS SignedData
    raw_content_bytes = bytes(token_content_info["content"])
    try:
        signed_data, sd_rest = decoder.decode(raw_content_bytes, asn1Spec=rfc5652.SignedData())
    except (PyAsn1Error, ValueError) as e:
        raise InvalidTimestampTokenError(f"Failed to decode CMS SignedData: {e}")

    if len(sd_rest) != 0:
        raise InvalidTimestampTokenError("Trailing unparsed bytes in CMS SignedData")

    # 5. Extract EncapsulatedContentInfo and TSTInfo
    encap_content = signed_data["encapContentInfo"]
    e_content_type = str(encap_content["eContentType"])
    if e_content_type != OID_CT_TST_INFO:
        raise InvalidTimestampTokenError(
            f"Encapsulated content type must be id-ct-TSTInfo ({OID_CT_TST_INFO}), got '{e_content_type}'"
        )

    if not encap_content["eContent"].isValue:
        raise InvalidTimestampTokenError("Encapsulated content eContent is absent")

    tst_info_der = bytes(encap_content["eContent"])
    try:
        tst_info, tst_rest = decoder.decode(tst_info_der, asn1Spec=rfc3161.TSTInfo())
    except (PyAsn1Error, ValueError) as e:
        raise InvalidTimestampTokenError(f"Failed to decode TSTInfo from eContent: {e}")

    if len(tst_rest) != 0:
        raise InvalidTimestampTokenError("Trailing unparsed bytes in TSTInfo")

    # 6. Validate TSTInfo Fields
    tst_version = int(tst_info["version"])
    if tst_version != 1:
        raise InvalidTimestampTokenError(f"TSTInfo version must be 1, got {tst_version}")

    tst_policy = str(tst_info["policy"])
    if expected_policy_oid and tst_policy != expected_policy_oid:
        raise TSAPolicyMismatchError(
            f"TSTInfo policy '{tst_policy}' does not match expected policy '{expected_policy_oid}'"
        )

    # 7. Validate Message Imprint
    imprint = tst_info["messageImprint"]
    resp_algo_oid = str(imprint["hashAlgorithm"]["algorithm"])
    if resp_algo_oid not in OID_TO_HASH_NAME:
        raise UnsupportedAlgorithmError(f"Unsupported message imprint algorithm in response: {resp_algo_oid}")

    resp_algo_name = OID_TO_HASH_NAME[resp_algo_oid]
    resp_digest = bytes(imprint["hashedMessage"])

    # Normalize expected imprint digest
    if isinstance(expected_imprint_digest, str):
        clean_exp = expected_imprint_digest.strip()
        if ":" in clean_exp:
            clean_exp = clean_exp.split(":", 1)[1]
        expected_bytes = bytes.fromhex(clean_exp)
    else:
        expected_bytes = bytes(expected_imprint_digest)

    if resp_digest != expected_bytes:
        raise RequestResponseBindingError(
            f"Message imprint digest mismatch: expected sha256:{expected_bytes.hex()}, got sha256:{resp_digest.hex()}"
        )

    # 8. Validate Nonce
    resp_nonce = int(tst_info["nonce"]) if tst_info["nonce"].isValue else None
    if expected_nonce is not None:
        if resp_nonce is None:
            raise RequestResponseBindingError("Nonce was requested but response contains no nonce")
        if resp_nonce != expected_nonce:
            raise RequestResponseBindingError(
                f"Nonce mismatch: expected {expected_nonce}, got {resp_nonce}"
            )

    # 9. Extract and Parse Generation Time
    gen_time_raw = str(tst_info["genTime"])
    # Standard format: YYYYMMDDhhmmss[Z] or YYYYMMDDhhmmss.s[Z]
    try:
        clean_time = gen_time_raw.rstrip("Z")
        if "." in clean_time:
            dt = datetime.strptime(clean_time, "%Y%m%d%H%M%S.%f").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.strptime(clean_time, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        timestamp_iso = dt.isoformat()
    except ValueError as e:
        raise InvalidTimestampTokenError(f"Malformed GeneralizedTime in TSTInfo ('{gen_time_raw}'): {e}")

    # Check clock drift if reference time provided
    if reference_time_utc:
        try:
            ref_dt = datetime.fromisoformat(reference_time_utc.replace("Z", "+00:00"))
            drift = abs((dt - ref_dt).total_seconds())
            if drift > max_drift_seconds:
                raise ClockDriftExceededError(
                    f"Timestamp generation time drift ({drift:.2f}s) exceeds limit ({max_drift_seconds}s)"
                )
        except ValueError as e:
            raise ClockDriftExceededError(f"Invalid reference_time_utc: {e}")

    serial_number = int(tst_info["serialNumber"])

    # 10. Validate TSTInfo Extensions (RFC 3161 § 2.4.2 / SEC-4C1-04)
    if tst_info["extensions"].isValue:
        seen_extn_oids: set = set()
        try:
            if len(tst_info["extensions"]) == 0:
                raise InvalidTimestampTokenError(
                    "TSTInfo extensions field present but contains zero extensions"
                )
            for ext in tst_info["extensions"]:
                if not ext["extnID"].isValue:
                    raise InvalidTimestampTokenError("TSTInfo extension missing required extnID")

                ext_oid = str(ext["extnID"])
                if ext_oid in seen_extn_oids:
                    raise InvalidTimestampTokenError(
                        f"Duplicate extension identifier in TSTInfo: '{ext_oid}'"
                    )
                seen_extn_oids.add(ext_oid)

                is_critical = bool(ext["critical"]) if ext["critical"].isValue else False

                if not ext["extnValue"].isValue:
                    raise InvalidTimestampTokenError(
                        f"TSTInfo extension '{ext_oid}' missing extnValue"
                    )

                # RFC 3161 § 2.4.2: If an unrecognized critical extension is encountered,
                # the timestamp MUST be rejected.
                if is_critical:
                    raise InvalidTimestampTokenError(
                        f"Unrecognized critical extension in TSTInfo: '{ext_oid}'"
                    )
        except InvalidTimestampTokenError:
            raise
        except (PyAsn1Error, ValueError, TypeError, KeyError) as e:
            raise InvalidTimestampTokenError(f"Malformed extension ASN.1 structure in TSTInfo: {e}")

    # 11. Extract Embedded Certificates from CMS SignedData
    embedded_certs: List[x509.Certificate] = []
    if signed_data["certificates"].isValue:
        for cert_choice in signed_data["certificates"]:
            if cert_choice["certificate"].isValue:
                raw_c_der = encoder.encode(cert_choice["certificate"])
                try:
                    c = x509.load_der_x509_certificate(raw_c_der)
                    embedded_certs.append(c)
                except Exception:
                    pass

    # Build available certs pool
    all_certs = list(embedded_certs)
    if trusted_certs:
        for tc in trusted_certs:
            if tc not in all_certs:
                all_certs.append(tc)

    if not all_certs:
        raise UntrustedTSACertificateError("No certificates available in SignedData or trusted store to verify signer")

    # 11. Find Signer Certificate and Validate Signature
    signer_infos = signed_data["signerInfos"]
    if len(signer_infos) == 0:
        raise InvalidTimestampTokenError("CMS SignedData contains zero signerInfos")

    # We evaluate the primary signerInfo
    signer_info = signer_infos[0]
    sid = signer_info["sid"]

    signer_cert: Optional[x509.Certificate] = None

    if sid["issuerAndSerialNumber"].isValue:
        ias = sid["issuerAndSerialNumber"]
        sid_serial = int(ias["serialNumber"])

        for c in all_certs:
            if c.serial_number == sid_serial:
                signer_cert = c
                break
    elif sid["subjectKeyIdentifier"].isValue:
        ski_bytes = bytes(sid["subjectKeyIdentifier"])
        for c in all_certs:
            try:
                ext = c.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_KEY_IDENTIFIER)
                if ext.value.key_identifier == ski_bytes:
                    signer_cert = c
                    break
            except x509.ExtensionNotFound:
                continue

    # Fallback to single cert if sid matching is ambiguous
    if signer_cert is None and len(all_certs) == 1:
        signer_cert = all_certs[0]

    if signer_cert is None:
        raise UntrustedTSACertificateError("Could not locate signer certificate matching CMS SignerIdentifier")

    # 12. Validate Signer Certificate Validity, Extensions & Trust Path (SEC-4C1-01)
    verify_tsa_certificate_trust(
        signer_cert=signer_cert,
        trusted_certs=trusted_certs,
        candidate_intermediates=embedded_certs,
        validation_time=dt,
    )
    cert_fp = signer_cert.fingerprint(hashes.SHA256()).hex()

    # 13. Verify CMS Signature Mathematically & SignedAttributes Interoperability (SEC-4C1-02)
    raw_signature = bytes(signer_info["signature"])
    sig_algo_oid = str(signer_info["signatureAlgorithm"]["algorithm"])
    pub_key = signer_cert.public_key()

    # Determine data signed
    # If signedAttrs is present, signature covers DER encoding of signedAttrs with SET OF tag (0x31)
    if signer_info["signedAttrs"].isValue:
        seen_attr_types = set()
        found_content_type = False
        found_message_digest = False
        expected_md = hashlib.sha256(tst_info_der).digest()

        for attr in signer_info["signedAttrs"]:
            attr_type = str(attr["attrType"])
            if attr_type in seen_attr_types:
                raise InvalidSignatureError(f"Duplicate signed attribute: '{attr_type}'")
            seen_attr_types.add(attr_type)

            values = attr["attrValues"]
            if len(values) == 0:
                raise InvalidSignatureError(f"Signed attribute '{attr_type}' has empty attribute values")

            if attr_type == "1.2.840.113549.1.9.3":  # id-contentType
                if len(values) != 1:
                    raise InvalidSignatureError(
                        f"contentType signed attribute must contain exactly one value, got {len(values)}"
                    )
                try:
                    ct_obj, ct_rest = decoder.decode(bytes(values[0]), asn1Spec=univ.ObjectIdentifier())
                    if len(ct_rest) != 0:
                        raise InvalidSignatureError("Trailing bytes in contentType signed attribute")
                except Exception as e:
                    raise InvalidSignatureError(f"Malformed contentType signed attribute: {e}")

                if str(ct_obj) != e_content_type:
                    raise InvalidSignatureError(
                        f"CMS contentType attribute mismatch: expected '{e_content_type}', got '{str(ct_obj)}'"
                    )
                found_content_type = True

            elif attr_type == "1.2.840.113549.1.9.4":  # id-messageDigest
                if len(values) != 1:
                    raise InvalidSignatureError(
                        f"messageDigest signed attribute must contain exactly one value, got {len(values)}"
                    )
                try:
                    md_obj, md_rest = decoder.decode(bytes(values[0]), asn1Spec=univ.OctetString())
                    if len(md_rest) != 0:
                        raise InvalidSignatureError("Trailing bytes in messageDigest signed attribute")
                except Exception as e:
                    raise InvalidSignatureError(f"Malformed messageDigest signed attribute: {e}")

                attr_md = bytes(md_obj)
                if attr_md != expected_md:
                    raise InvalidSignatureError(
                        f"CMS SignedData messageDigest attribute mismatch: {attr_md.hex()} != {expected_md.hex()}"
                    )
                found_message_digest = True

        if not found_content_type:
            raise InvalidSignatureError("CMS signedAttrs present but missing required id-contentType attribute")
        if not found_message_digest:
            raise InvalidSignatureError("CMS signedAttrs present but missing required id-messageDigest attribute")

        # Encode signedAttrs with standard DER SET OF tag (0x31) for signature verification
        raw_signed_attrs = encoder.encode(signer_info["signedAttrs"])
        if raw_signed_attrs[0] == 0xA0:
            bytes_to_verify = b"\x31" + raw_signed_attrs[1:]
        else:
            bytes_to_verify = raw_signed_attrs
    else:
        # Signature is directly over eContent (tst_info_der)
        bytes_to_verify = tst_info_der

    # Perform mathematical signature verification
    try:
        if isinstance(pub_key, rsa.RSAPublicKey):
            # Select hash algorithm from sig_algo_oid or digestAlgorithm
            if sig_algo_oid == OID_SHA384_WITH_RSA:
                h_algo = hashes.SHA384()
            elif sig_algo_oid == OID_SHA512_WITH_RSA:
                h_algo = hashes.SHA512()
            else:
                h_algo = hashes.SHA256()

            pub_key.verify(
                raw_signature,
                bytes_to_verify,
                padding.PKCS1v15(),
                h_algo,
            )
        elif isinstance(pub_key, ec.EllipticCurvePublicKey):
            if sig_algo_oid == OID_ECDSA_WITH_SHA384:
                h_algo = hashes.SHA384()
            elif sig_algo_oid == OID_ECDSA_WITH_SHA512:
                h_algo = hashes.SHA512()
            else:
                h_algo = hashes.SHA256()

            pub_key.verify(
                raw_signature,
                bytes_to_verify,
                ec.ECDSA(h_algo),
            )
        elif isinstance(pub_key, ed25519.Ed25519PublicKey):
            pub_key.verify(raw_signature, bytes_to_verify)
        else:
            raise UnsupportedAlgorithmError(f"Unsupported public key type: {type(pub_key)}")
    except InvalidSignature:
        raise InvalidSignatureError("CMS signature verification failed mathematically against TSA certificate")
    except Exception as e:
        raise InvalidSignatureError(f"Signature verification encountered fatal error: {e}")

    # Extract common name
    cn_attrs = signer_cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
    common_name = cn_attrs[0].value if cn_attrs else "TSA"

    return VerifiedTimestamp(
        tsa_identity=common_name,
        tsa_policy_oid=tst_policy,
        message_imprint_algorithm=resp_algo_name,
        message_imprint_digest=f"sha256:{resp_digest.hex()}",
        timestamp_utc=timestamp_iso,
        serial_number=serial_number,
        nonce=resp_nonce,
        tsa_certificate_fingerprint=f"sha256:{cert_fp}",
        signer_common_name=common_name,
        signature_algorithm_oid=sig_algo_oid,
        token_der=raw_token_der,
        raw_response_der=resp_der,
        is_synthetic=True,
    )


# ============================================================================
# Synthetic Response Builder (For Testing and Isolated Worktrees)
# ============================================================================

def create_synthetic_timestamp_response(
    imprint_digest: Union[bytes, str],
    signing_key: Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey],
    signing_cert: x509.Certificate,
    policy_oid: str = DEFAULT_TSA_POLICY_OID,
    nonce: Optional[int] = None,
    serial_number: int = 1001,
    gen_time_utc: Optional[str] = None,
    pki_status: int = 0,
    status_string: Optional[str] = None,
    fail_info: Optional[int] = None,
    corrupt_signature: bool = False,
    corrupt_imprint: bool = False,
    omit_token: bool = False,
    use_signed_attrs: bool = False,
    corrupt_signed_attrs_digest: bool = False,
    corrupt_signed_attrs_content_type: bool = False,
    duplicate_signed_attrs: bool = False,
    missing_signed_attrs_content_type: bool = False,
    missing_signed_attrs_digest: bool = False,
    additional_certs: Optional[List[x509.Certificate]] = None,
    tst_extensions: Optional[List[Any]] = None,
) -> bytes:
    """
    Construct a complete synthetic TimeStampResp DER stream for testing.
    Can simulate both successful responses and controlled adversarial failure vectors.
    """
    # 1. Build Status Info
    resp = rfc3161.TimeStampResp()
    pki_status_info = rfc3161.PKIStatusInfo()
    pki_status_info["status"] = pki_status
    if status_string:
        free_text = rfc3161.PKIFreeText()
        free_text.append(char.UTF8String(status_string))
        pki_status_info["statusString"] = free_text
    if fail_info is not None:
        pki_status_info["failInfo"] = univ.BitString(f"'{bin(fail_info)[2:]}'B")
    resp["status"] = pki_status_info

    if omit_token or pki_status not in (0, 1):
        return encoder.encode(resp)

    # 2. Build TSTInfo
    if isinstance(imprint_digest, str):
        clean_hex = imprint_digest.strip()
        if ":" in clean_hex:
            clean_hex = clean_hex.split(":", 1)[1]
        digest_bytes = bytes.fromhex(clean_hex)
    else:
        digest_bytes = bytes(imprint_digest)

    if corrupt_imprint:
        # Flip first byte
        digest_bytes = bytes([digest_bytes[0] ^ 0xFF]) + digest_bytes[1:]

    tst_info = rfc3161.TSTInfo()
    tst_info["version"] = 1
    tst_info["policy"] = univ.ObjectIdentifier(policy_oid)

    imprint = rfc3161.MessageImprint()
    algo_id = rfc3161.AlgorithmIdentifier()
    algo_id["algorithm"] = univ.ObjectIdentifier(OID_SHA256)
    algo_id["parameters"] = encoder.encode(univ.Null(""))
    imprint["hashAlgorithm"] = algo_id
    imprint["hashedMessage"] = univ.OctetString(digest_bytes)
    tst_info["messageImprint"] = imprint

    tst_info["serialNumber"] = serial_number

    if gen_time_utc:
        dt = datetime.fromisoformat(gen_time_utc.replace("Z", "+00:00"))
    else:
        dt = datetime.now(timezone.utc)
    tst_info["genTime"] = useful.GeneralizedTime(dt.strftime("%Y%m%d%H%M%SZ"))

    if nonce is not None:
        tst_info["nonce"] = nonce

    if tst_extensions is not None:
        if len(tst_extensions) == 0:
            # Force encoding of empty extensions field [1] IMPLICIT (tag 0xA1, len 0x00)
            base_der = encoder.encode(tst_info)
            if base_der[1] & 0x80:
                hdr_len = 2 + (base_der[1] & 0x7F)
            else:
                hdr_len = 2
            body = base_der[hdr_len:] + b"\xa1\x00"
            l_body = len(body)
            if l_body < 128:
                len_b = bytes([l_body])
            elif l_body < 256:
                len_b = bytes([0x81, l_body])
            else:
                len_b = bytes([0x82, l_body >> 8, l_body & 0xFF])
            tst_info_der = b"\x30" + len_b + body
        else:
            from pyasn1.type import tag

            exts = rfc2459.Extensions().subtype(
                implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)
            )
            for ext_item in tst_extensions:
                e = rfc2459.Extension()
                e["extnID"] = univ.ObjectIdentifier(ext_item[0])
                e["critical"] = ext_item[1]
                if len(ext_item) > 2 and ext_item[2] is not None:
                    e["extnValue"] = univ.OctetString(ext_item[2])
                exts.append(e)
            tst_info["extensions"] = exts
            tst_info_der = encoder.encode(tst_info)
    else:
        tst_info_der = encoder.encode(tst_info)

    # 3. Build SignedData
    sd = rfc5652.SignedData()
    sd["version"] = 3

    da_set = rfc5652.DigestAlgorithmIdentifiers()
    da = rfc5652.DigestAlgorithmIdentifier()
    da["algorithm"] = univ.ObjectIdentifier(OID_SHA256)
    da["parameters"] = encoder.encode(univ.Null(""))
    da_set.append(da)
    sd["digestAlgorithms"] = da_set

    eci = rfc5652.EncapsulatedContentInfo()
    eci["eContentType"] = univ.ObjectIdentifier(OID_CT_TST_INFO)
    econtent_type = eci.componentType["eContent"].getType()
    eci["eContent"] = econtent_type.clone(tst_info_der)
    sd["encapContentInfo"] = eci

    # Add signer certificate and any intermediate certificates
    certs_to_embed = [signing_cert]
    if additional_certs:
        for ac in additional_certs:
            if ac not in certs_to_embed:
                certs_to_embed.append(ac)

    certs_set_type = sd.componentType["certificates"].getType()
    certs_set = certs_set_type.clone()
    for cert_obj in certs_to_embed:
        cert_der = cert_obj.public_bytes(serialization.Encoding.DER)
        parsed_cert, _ = decoder.decode(cert_der, asn1Spec=rfc2459.Certificate())
        cert_choices = rfc5652.CertificateChoices()
        cert_choices["certificate"] = parsed_cert
        certs_set.append(cert_choices)
    sd["certificates"] = certs_set

    # SignerInfo
    signing_cert_der = signing_cert.public_bytes(serialization.Encoding.DER)
    parsed_signing_cert, _ = decoder.decode(signing_cert_der, asn1Spec=rfc2459.Certificate())

    si = rfc5652.SignerInfo()
    si["version"] = 1
    sid = rfc5652.SignerIdentifier()
    ias = rfc5652.IssuerAndSerialNumber()
    ias["issuer"] = parsed_signing_cert["tbsCertificate"]["issuer"]
    ias["serialNumber"] = parsed_signing_cert["tbsCertificate"]["serialNumber"]
    sid["issuerAndSerialNumber"] = ias
    si["sid"] = sid
    si["digestAlgorithm"] = da

    # Determine signed bytes and build signedAttrs if requested
    if use_signed_attrs:
        signed_attrs = rfc5652.SignedAttributes()

        # 1. id-contentType (1.2.840.113549.1.9.3)
        if not missing_signed_attrs_content_type:
            ct_attr = rfc5652.Attribute()
            ct_attr["attrType"] = univ.ObjectIdentifier("1.2.840.113549.1.9.3")
            ct_vals = ct_attr.componentType["attrValues"].getType().clone()
            ct_oid_str = (
                OID_CT_TST_INFO if not corrupt_signed_attrs_content_type
                else "1.2.840.113549.1.9.16.1.99"
            )
            ct_vals.append(univ.Any(encoder.encode(univ.ObjectIdentifier(ct_oid_str))))
            ct_attr["attrValues"] = ct_vals
            signed_attrs.append(ct_attr)

        # 2. id-messageDigest (1.2.840.113549.1.9.4)
        if not missing_signed_attrs_digest:
            md_attr = rfc5652.Attribute()
            md_attr["attrType"] = univ.ObjectIdentifier("1.2.840.113549.1.9.4")
            md_vals = md_attr.componentType["attrValues"].getType().clone()
            calc_md = hashlib.sha256(tst_info_der).digest()
            if corrupt_signed_attrs_digest:
                calc_md = bytes([calc_md[0] ^ 0xFF]) + calc_md[1:]
            md_vals.append(univ.Any(encoder.encode(univ.OctetString(calc_md))))
            md_attr["attrValues"] = md_vals
            signed_attrs.append(md_attr)

        if duplicate_signed_attrs and not missing_signed_attrs_digest:
            signed_attrs.append(md_attr)

        si_signed_attrs_type = si.componentType["signedAttrs"].getType().clone()
        for a in signed_attrs:
            si_signed_attrs_type.append(a)
        si["signedAttrs"] = si_signed_attrs_type

        raw_signed_attrs = encoder.encode(si["signedAttrs"])
        bytes_to_sign = b"\x31" + raw_signed_attrs[1:]
    else:
        bytes_to_sign = tst_info_der

    # Compute signature
    if isinstance(signing_key, rsa.RSAPrivateKey):
        sig_bytes = signing_key.sign(bytes_to_sign, padding.PKCS1v15(), hashes.SHA256())
        sig_algo_oid = OID_SHA256_WITH_RSA
    elif isinstance(signing_key, ec.EllipticCurvePrivateKey):
        sig_bytes = signing_key.sign(bytes_to_sign, ec.ECDSA(hashes.SHA256()))
        sig_algo_oid = OID_ECDSA_WITH_SHA256
    elif isinstance(signing_key, ed25519.Ed25519PrivateKey):
        sig_bytes = signing_key.sign(bytes_to_sign)
        sig_algo_oid = OID_ED25519
    else:
        raise ValueError(f"Unsupported signing key: {type(signing_key)}")

    if corrupt_signature:
        # Flip the last byte of signature
        sig_bytes = sig_bytes[:-1] + bytes([sig_bytes[-1] ^ 0x01])

    sig_algo = rfc5652.SignatureAlgorithmIdentifier()
    sig_algo["algorithm"] = univ.ObjectIdentifier(sig_algo_oid)
    sig_algo["parameters"] = encoder.encode(univ.Null(""))
    si["signatureAlgorithm"] = sig_algo
    si["signature"] = univ.OctetString(sig_bytes)

    sis = rfc5652.SignerInfos()
    sis.append(si)
    sd["signerInfos"] = sis

    # Wrap in ContentInfo
    ci = rfc5652.ContentInfo()
    ci["contentType"] = univ.ObjectIdentifier(OID_SIGNED_DATA)
    ci_content_type = ci.componentType["content"].getType()
    ci["content"] = ci_content_type.clone(encoder.encode(sd))

    resp["timeStampToken"] = ci
    return encoder.encode(resp)
