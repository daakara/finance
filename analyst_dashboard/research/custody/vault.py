"""
ARX Terminal — Local Restricted Evidence Vault Implementation (Gate C)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Operational Mandate:
- Configurable external storage location via ARX_CUSTODY_VAULT_ROOT or parameter.
- Fail-closed refusal if unset or if pointing inside the Git repository tree.
- Native NTFS ACL enforcement & verification on Windows (POSIX 0o700 on Linux/macOS).
- Strict separation of sealed evidence, identities, staging, and receipts.
- Atomic write semantics: staging write -> flush -> fsync -> atomic rename.
- Overwrite prevention: sealed evidence cannot be replaced or overwritten.
- Cryptographic shredding on participant withdrawal / retention purge.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

from analyst_dashboard.research.custody.exceptions import (
    CustodyError,
    VaultConfigurationError,
    VaultPermissionError,
)


def find_git_repository_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the root of the containing Git repository by walking parent directories."""
    curr = (start_path or Path.cwd()).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists():
            return parent
    return None


class LocalRestrictedVault:
    """
    Isolated local evidence vault strictly located outside ordinary Git repositories.
    Enforces native OS permissions, atomic storage, and partition separation.
    """

    SUBDIR_SEALED = "sealed"
    SUBDIR_STAGING = "staging"
    SUBDIR_IDENTITIES = "identities"
    SUBDIR_LEDGER = "ledger"

    def __init__(
        self,
        storage_root: Optional[str | Path] = None,
        enforce_acls: bool = True,
        strict_permissions_check: bool = True,
    ):
        """
        Initialize vault at specified root or from ARX_CUSTODY_VAULT_ROOT env var.
        Fails closed if root is unset, empty, or located inside Git repository.
        """
        raw_root = storage_root or os.environ.get("ARX_CUSTODY_VAULT_ROOT")
        if not raw_root:
            raise VaultConfigurationError(
                "ARX_CUSTODY_VAULT_ROOT is unset or empty. "
                "Fail-closed: local restricted vault requires an explicit external storage path."
            )

        self.root_path = Path(raw_root).resolve()

        # Git repository containment verification
        repo_root = find_git_repository_root(Path(__file__).resolve().parent)
        if repo_root:
            resolved_repo = repo_root.resolve()
            # Check if vault root is inside repository
            try:
                self.root_path.relative_to(resolved_repo)
                is_inside = True
            except ValueError:
                is_inside = False

            if is_inside or self.root_path == resolved_repo:
                raise VaultConfigurationError(
                    f"Configured vault root '{self.root_path}' is inside Git repository '{resolved_repo}'. "
                    f"Forbidden under ARX custody policy: sensitive research vaults must reside off-Git."
                )

        # Initialize subdirectories
        self.sealed_dir = self.root_path / self.SUBDIR_SEALED
        self.staging_dir = self.root_path / self.SUBDIR_STAGING
        self.identities_dir = self.root_path / self.SUBDIR_IDENTITIES
        self.ledger_dir = self.root_path / self.SUBDIR_LEDGER

        for directory in [
            self.root_path,
            self.sealed_dir,
            self.staging_dir,
            self.identities_dir,
            self.ledger_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        if enforce_acls:
            self.enforce_native_permissions()

        if strict_permissions_check:
            is_secure, reason = self.verify_permissions()
            if not is_secure:
                raise VaultPermissionError(
                    f"Vault root permissions verification failed: {reason}"
                )

    def enforce_native_permissions(self) -> None:
        """Enforce strict native OS access control restrictions on the vault directory."""
        if os.name == "nt":
            # Windows NTFS ACL enforcement: remove inheritance and grant only current user
            username = os.environ.get("USERNAME")
            if username:
                try:
                    grant_arg = f"{username}:(OI)(CI)F"
                    subprocess.run(
                        [
                            "icacls",
                            str(self.root_path),
                            "/inheritance:r",
                            "/grant:r",
                            grant_arg,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except Exception as e:
                    # In some sandbox environments icacls may return warnings
                    pass
        else:
            # POSIX permission enforcement: 0o700 (owner only)
            try:
                self.root_path.chmod(0o700)
                for d in [
                    self.sealed_dir,
                    self.staging_dir,
                    self.identities_dir,
                    self.ledger_dir,
                ]:
                    d.chmod(0o700)
            except Exception:
                pass

    def verify_permissions(self) -> Tuple[bool, str]:
        """Verify that filesystem permissions restrict access to authorized owner only."""
        if os.name == "nt":
            try:
                res = subprocess.run(
                    ["icacls", str(self.root_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                output = res.stdout
                path_str = str(self.root_path)
                acl_entries = []
                for line in output.splitlines():
                    if "Successfully processed" in line or not line.strip():
                        continue
                    entry = line.replace(path_str, "").strip()
                    if entry:
                        acl_entries.append(entry)

                if not acl_entries:
                    return False, "No ACL entries found for vault root."

                # Check for inherited permissions or forbidden principals
                forbidden_principals = ["Everyone", "BUILTIN\\Users", "Authenticated Users"]
                for entry in acl_entries:
                    if "(I)" in entry:
                        return False, f"Inherited permissions detected in entry: '{entry}'. Inheritance must be disabled."
                    for fp in forbidden_principals:
                        if fp.lower() in entry.lower():
                            return False, f"Forbidden principal '{fp}' has access in ACL entry: '{entry}'."

                return True, "NTFS ACL restricted to authorized owner."
            except Exception as e:
                return False, f"Failed to execute icacls check: {e}"
        else:
            try:
                mode = self.root_path.stat().st_mode
                if (mode & 0o077) != 0:
                    return False, f"POSIX permissions unsafe ({oct(mode)}). Must be 0o700."
                return True, "POSIX permissions verified (0o700)."
            except Exception as e:
                return False, f"Failed to check POSIX mode: {e}"

    def store_sealed_evidence(self, submission_id: str, ciphertext: bytes) -> Path:
        """
        Store encrypted submission using atomic staging write and rename.
        Refuses to overwrite existing sealed evidence.
        """
        if not submission_id or not isinstance(submission_id, str):
            raise CustodyError("Invalid submission_id for sealed evidence storage.")

        target_file = self.sealed_dir / f"{submission_id}.enc"
        if target_file.exists():
            raise CustodyError(
                f"Sealed evidence '{submission_id}' already exists. Overwrite prohibited."
            )

        tmp_file = self.staging_dir / f"{submission_id}.tmp"
        try:
            with open(tmp_file, "wb") as f:
                f.write(ciphertext)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace/rename
            os.replace(tmp_file, target_file)
            return target_file
        except Exception as e:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
            raise CustodyError(f"Failed to atomically store sealed evidence: {e}") from e

    def read_sealed_evidence(self, submission_id: str) -> bytes:
        """Read raw ciphertext for specified submission identifier."""
        target_file = self.sealed_dir / f"{submission_id}.enc"
        if not target_file.exists():
            raise CustodyError(f"Sealed evidence '{submission_id}' not found in vault.")
        return target_file.read_bytes()

    def purge_sealed_evidence(self, submission_id: str) -> bool:
        """Cryptographically shred and delete sealed submission file."""
        target_file = self.sealed_dir / f"{submission_id}.enc"
        if not target_file.exists():
            return False

        file_size = target_file.stat().st_size
        # Overwrite with random bytes
        with open(target_file, "wb") as f:
            f.write(os.urandom(file_size))
            f.flush()
            os.fsync(f.fileno())

        target_file.unlink()
        return True

    def store_identity_mapping(self, pseudonym: str, encrypted_identity: bytes) -> Path:
        """Store encrypted identity mapping in segregated identities directory."""
        target_file = self.identities_dir / f"{pseudonym}.id.enc"
        if target_file.exists():
            raise CustodyError(
                f"Identity mapping for '{pseudonym}' already exists. Overwrite prohibited."
            )

        tmp_file = self.staging_dir / f"{pseudonym}.id.tmp"
        with open(tmp_file, "wb") as f:
            f.write(encrypted_identity)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, target_file)
        return target_file

    def read_identity_mapping(self, pseudonym: str) -> Optional[bytes]:
        """Read encrypted identity mapping for pseudonym."""
        target_file = self.identities_dir / f"{pseudonym}.id.enc"
        if not target_file.exists():
            return None
        return target_file.read_bytes()

    def purge_identity_mapping(self, pseudonym: str) -> bool:
        """Cryptographically shred and delete identity mapping file."""
        target_file = self.identities_dir / f"{pseudonym}.id.enc"
        if not target_file.exists():
            return False

        file_size = target_file.stat().st_size
        with open(target_file, "wb") as f:
            f.write(os.urandom(file_size))
            f.flush()
            os.fsync(f.fileno())

        target_file.unlink()
        return True
