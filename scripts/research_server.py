"""
ARX Terminal — Local Loopback Research Server & Custody Intake
Defect: ARX-UX-P00-INSTR-001
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Governing Authorization: RO-DEC-2026-10-23-012

Operational Mandate:
- Binds strictly to 127.0.0.1 (zero LAN/public network exposure).
- Serves static Next.js prototype export from prototype-harness/out.
- Enforces Host and Origin validation against loopback interface.
- Real authentication boundary beyond same-origin: moderator-controlled bootstrap token.
- Issues rotating in-memory session tokens with bounded 2-hour TTL.
- Enforces Bearer token authentication on all custody write and readback endpoints.
- Endpoints under /api/research/custody/:
    POST /api/research/custody/handshake
    POST /api/research/custody/session_ingest
    POST /api/research/custody/verify_receipt
    GET  /api/research/custody/receipt
    POST /api/research/custody/withdraw
"""

import os
import sys
import json
import time
import secrets
import argparse
import tempfile
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

# Ensure finance repository is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.research.custody.vault import LocalRestrictedVault
from analyst_dashboard.research.custody.crypto import CustodyCrypto, generate_key
from analyst_dashboard.research.custody.ledger import TamperEvidentLedger
from analyst_dashboard.research.custody.session_intake import (
    ParticipantSessionIntakePipeline,
    CustodyConflictError,
    IntakeValidationError,
    CustodyError,
)

MAX_BODY_BYTES = 2 * 1024 * 1024  # 2MB ceiling


class ResearchServerState:
    """Thread-safe server state holding custody pipeline and active session tokens."""

    def __init__(
        self,
        pipeline: ParticipantSessionIntakePipeline,
        bootstrap_token: str,
        harness_dir: Path,
        allowed_port: int,
    ):
        self.pipeline = pipeline
        self.bootstrap_token = bootstrap_token
        self.harness_dir = harness_dir
        self.allowed_port = allowed_port
        self.session_tokens: Dict[str, Dict[str, Any]] = {}

    def issue_session_token(self, session_id: str, participant_id: str) -> str:
        token = secrets.token_hex(32)
        now = time.time()
        self.session_tokens[token] = {
            "session_id": session_id,
            "participant_id": participant_id,
            "created_at": now,
            "expires_at": now + 7200,  # 2 hours
        }
        return token

    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        info = self.session_tokens.get(token)
        if not info:
            return None
        if time.time() > info["expires_at"]:
            del self.session_tokens[token]
            return None
        return info

    def invalidate_session_token(self, token: str) -> None:
        if token in self.session_tokens:
            del self.session_tokens[token]


class ResearchRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP request handler serving static prototype assets and restricted custody API.
    """

    server_state: ResearchServerState

    def __init__(self, *args, **kwargs):
        # Initialize SimpleHTTPRequestHandler with harness directory
        super().__init__(*args, directory=str(self.server_state.harness_dir), **kwargs)

    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._add_security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _add_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; connect-src 'self';",
        )

    def end_headers(self) -> None:
        # Override to add security headers on static GET responses as well
        if not hasattr(self, "_security_headers_added"):
            self._add_security_headers()
            self._security_headers_added = True
        super().end_headers()

    def _verify_request_boundary(self) -> bool:
        """Enforce strict Host, Origin, and Sec-Fetch-Site loopback restrictions."""
        host = self.headers.get("Host", "")
        allowed_hosts = {
            f"127.0.0.1:{self.server_state.allowed_port}",
            f"localhost:{self.server_state.allowed_port}",
        }
        if host not in allowed_hosts:
            self._send_json_response(403, {"error": "FORBIDDEN_INVALID_HOST", "host": host})
            return False

        origin = self.headers.get("Origin")
        if origin:
            allowed_origins = {
                f"http://127.0.0.1:{self.server_state.allowed_port}",
                f"http://localhost:{self.server_state.allowed_port}",
            }
            if origin not in allowed_origins:
                self._send_json_response(403, {"error": "FORBIDDEN_INVALID_ORIGIN", "origin": origin})
                return False

        sec_fetch_site = self.headers.get("Sec-Fetch-Site")
        if sec_fetch_site and sec_fetch_site == "cross-site":
            self._send_json_response(403, {"error": "FORBIDDEN_CROSS_SITE_REQUEST"})
            return False

        return True

    def _extract_bearer_token(self) -> Optional[str]:
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer ") :].strip()
        return None

    def do_GET(self) -> None:
        if not self._verify_request_boundary():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        # Handle Custody Receipt Readback
        if path == "/api/research/custody/receipt":
            token = self._extract_bearer_token()
            session_info = self.server_state.validate_session_token(token) if token else None
            if not session_info:
                self._send_json_response(401, {"error": "UNAUTHORIZED_RESEARCH_SESSION"})
                return

            params = parse_qs(parsed.query)
            receipt_id = params.get("receipt_id", [None])[0]
            session_id = params.get("session_id", [None])[0]

            if not receipt_id or not session_id:
                self._send_json_response(400, {"error": "MISSING_RECEIPT_OR_SESSION_ID"})
                return

            # Lookup in ledger
            all_rcpts = self.server_state.pipeline.ledger.get_all_receipts()
            found = None
            for r in all_rcpts:
                if r.get("receipt_id") == receipt_id and r.get("metadata", {}).get("session_id") == session_id:
                    found = r
                    break

            if not found:
                self._send_json_response(404, {"error": "RECEIPT_NOT_FOUND"})
                return

            self._send_json_response(200, {
                "receipt_id": found["receipt_id"],
                "session_id": session_id,
                "timestamp_utc": found["timestamp_utc"],
                "custody_state": found["custody_state"],
                "ledger_ref": found["receipt_digest"],
                "source_payload_sha256": found["raw_submission_sha256"],
            })
            return

        # Fallback to static prototype serving
        super().do_GET()

    def do_POST(self) -> None:
        if not self._verify_request_boundary():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        # Reject any POST outside /api/research/custody/
        if not path.startswith("/api/research/custody/"):
            self._send_json_response(404, {"error": "ENDPOINT_NOT_FOUND", "path": path})
            return

        # Check content length
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json_response(400, {"error": "INVALID_CONTENT_LENGTH"})
            return

        if content_length > MAX_BODY_BYTES:
            self._send_json_response(413, {"error": "PAYLOAD_TOO_LARGE", "limit_bytes": MAX_BODY_BYTES})
            return

        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        # 1. Moderator Handshake Bootstrap Endpoint
        if path == "/api/research/custody/handshake":
            try:
                body_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                self._send_json_response(400, {"error": "INVALID_JSON_BODY"})
                return

            provided_bootstrap = body_data.get("bootstrap_token", "")
            session_id = body_data.get("session_id", f"SESS-{secrets.token_hex(6).upper()}")
            participant_id = body_data.get("participant_id", "P00-SYNTHETIC")

            # Check bootstrap token against server state
            if not secrets.compare_digest(provided_bootstrap, self.server_state.bootstrap_token):
                self._send_json_response(403, {"error": "UNAUTHORIZED_MODERATOR_BOOTSTRAP"})
                return

            session_token = self.server_state.issue_session_token(session_id, participant_id)
            self._send_json_response(200, {
                "status": "AUTHORIZED",
                "session_token": session_token,
                "expires_in": 7200,
                "session_id": session_id,
                "participant_id": participant_id,
            })
            return

        # Enforce Session Token Authorization on all other custody endpoints
        token = self._extract_bearer_token()
        session_info = self.server_state.validate_session_token(token) if token else None
        if not session_info:
            self._send_json_response(401, {"error": "UNAUTHORIZED_RESEARCH_SESSION"})
            return

        # 2. Custody Ingest Endpoint
        if path == "/api/research/custody/session_ingest":
            content_type = self.headers.get("Content-Type", "")
            if not ("application/x-jsonlines" in content_type or "application/json" in content_type):
                self._send_json_response(415, {"error": "UNSUPPORTED_MEDIA_TYPE", "required": "application/x-jsonlines"})
                return

            session_id = session_info["session_id"]
            try:
                receipt = self.server_state.pipeline.ingest_session_payload(
                    raw_bytes=raw_body,
                    session_id=session_id,
                )
                self._send_json_response(200, receipt)
            except CustodyConflictError as e:
                self._send_json_response(409, {"error": "SESSION_ALREADY_SEALED", "details": str(e)})
            except IntakeValidationError as e:
                self._send_json_response(422, {"error": "SCHEMA_VALIDATION_ERROR", "details": str(e)})
            except Exception as e:
                self._send_json_response(500, {"error": "CUSTODY_INTERNAL_ERROR", "details": str(e)})
            return

        # 3. Receipt Verification Endpoint
        if path == "/api/research/custody/verify_receipt":
            try:
                receipt_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                self._send_json_response(400, {"error": "INVALID_JSON_BODY"})
                return

            is_valid = self.server_state.pipeline.verify_server_receipt_signature(receipt_data)
            if is_valid:
                self._send_json_response(200, {
                    "verified": True,
                    "receipt_id": receipt_data.get("receipt_id"),
                    "ledger_ref": receipt_data.get("ledger_ref"),
                })
            else:
                self._send_json_response(400, {
                    "verified": False,
                    "error": "SERVER_RECEIPT_SIGNATURE_INVALID",
                })
            return

        # 4. Participant Withdrawal Endpoint
        if path == "/api/research/custody/withdraw":
            try:
                body_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                self._send_json_response(400, {"error": "INVALID_JSON_BODY"})
                return

            s_id = body_data.get("session_id", session_info["session_id"])
            p_id = body_data.get("participant_id", session_info["participant_id"])
            reason = body_data.get("reason", "PARTICIPANT_WITHDRAWAL")

            try:
                tombstone = self.server_state.pipeline.execute_session_withdrawal(
                    session_id=s_id,
                    participant_id=p_id,
                    withdrawal_reason=reason,
                )
                self.server_state.invalidate_session_token(token)
                self._send_json_response(200, tombstone)
            except Exception as e:
                self._send_json_response(500, {"error": "WITHDRAWAL_FAILED", "details": str(e)})
            return

        self._send_json_response(404, {"error": "UNKNOWN_CUSTODY_ENDPOINT", "path": path})


def create_research_server(
    port: int = 3456,
    bootstrap_token: Optional[str] = None,
    harness_dir: Optional[Path] = None,
    pipeline: Optional[ParticipantSessionIntakePipeline] = None,
) -> ThreadingHTTPServer:
    """Create and configure the ThreadingHTTPServer with research state."""
    h_dir = harness_dir or (REPO_ROOT / "prototype-harness" / "out")
    if not h_dir.exists():
        # Fallback to prototype-harness directory if out does not exist
        h_dir = REPO_ROOT / "prototype-harness"

    token = bootstrap_token or os.environ.get("ARX_RESEARCH_MODERATOR_TOKEN") or secrets.token_hex(32)

    if not pipeline:
        # Default initialization with isolated vault outside git
        vault_root = os.environ.get("ARX_CUSTODY_VAULT_ROOT")
        if not vault_root:
            vault_root = Path(tempfile.gettempdir()) / "arx_custody_vault"
        vault = LocalRestrictedVault(storage_root=vault_root, enforce_acls=False, strict_permissions_check=False)
        crypto_key = os.environ.get("ARX_CUSTODY_ENCRYPTION_KEY") or generate_key()
        crypto = CustodyCrypto(crypto_key)
        ledger = TamperEvidentLedger(vault.ledger_dir / "ledger.jsonl")
        pipeline = ParticipantSessionIntakePipeline(vault, crypto, ledger, is_synthetic_mode=True)

    state = ResearchServerState(
        pipeline=pipeline,
        bootstrap_token=token,
        harness_dir=h_dir,
        allowed_port=port,
    )

    class BoundHandler(ResearchRequestHandler):
        server_state = state

    server = ThreadingHTTPServer(("127.0.0.1", port), BoundHandler)
    server.research_state = state  # type: ignore[attr-defined]
    return server


def main():
    parser = argparse.ArgumentParser(description="ARX Terminal Local Research Server")
    parser.add_argument("--port", type=int, default=3456, help="Local loopback port (default: 3456)")
    parser.add_argument("--bootstrap-token", type=str, default=None, help="Moderator bootstrap token")
    parser.add_argument("--token-file", type=str, default=None, help="Path to write bootstrap token for launcher")
    args = parser.parse_args()

    token = args.bootstrap_token or os.environ.get("ARX_RESEARCH_MODERATOR_TOKEN") or secrets.token_hex(32)

    if args.token_file:
        token_path = Path(args.token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token, encoding="utf-8")

    server = create_research_server(port=args.port, bootstrap_token=token)

    print(f"[*] ARX Research Server active on http://127.0.0.1:{args.port}/prototype/")
    print(f"[*] MODERATOR BOOTSTRAP TOKEN: {token}")
    print("[*] Bound strictly to loopback 127.0.0.1. Press Ctrl+C to terminate.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down research server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
