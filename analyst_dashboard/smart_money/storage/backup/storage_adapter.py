"""
ARX Terminal — Smart Money Backup Storage Adapters.

Defines provider-neutral storage abstraction supporting:
1. Local off-volume directory storage (disaster recovery partition);
2. S3-compatible remote object storage (Cloudflare R2, AWS S3, Google Cloud Storage);
3. Automated SHA-256 integrity verification upon write and read.
"""

import os
import time
import shutil
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class BackupStorageAdapter(ABC):
    """Abstract storage adapter contract for PostgreSQL backups."""

    @abstractmethod
    def upload_file(self, local_path: str, remote_filename: str) -> Dict[str, Any]:
        """Uploads a file to the storage destination and returns verification receipt."""
        pass

    @abstractmethod
    def download_file(self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None) -> str:
        """Downloads a file from storage destination to local path, bounded by deadline_monotonic."""
        pass

    @abstractmethod
    def list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        """Lists all stored backup artifacts, bounded by deadline_monotonic."""
        pass

    @abstractmethod
    def delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        """Deletes a backup artifact from storage, bounded by deadline_monotonic."""
        pass

    @abstractmethod
    def get_storage_destination(self) -> str:
        """Returns human-readable representation of the destination."""
        pass

    @abstractmethod
    def is_external(self) -> bool:
        """Returns True if the destination is external off-host cloud storage."""
        pass

    @abstractmethod
    def get_backup_metadata(self, remote_filename: str) -> Optional[Dict[str, Any]]:
        """Returns metadata for remote artifact if present, else None."""
        pass


class LocalOffVolumeStorageAdapter(BackupStorageAdapter):
    """
    Stores backups in a dedicated off-volume directory outside the database volume.
    Used for local testing, staging, and environments lacking cloud S3 buckets.
    """

    def __init__(self, base_directory: str):
        self.base_directory = os.path.abspath(base_directory)
        os.makedirs(self.base_directory, exist_ok=True)
        self._custom_metadata: Dict[str, Dict[str, Any]] = {}

    def set_custom_metadata(self, remote_filename: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Allows testing custom, absent, or corrupted remote object metadata."""
        if metadata is None:
            self._custom_metadata.pop(remote_filename, None)
        else:
            self._custom_metadata[remote_filename] = metadata

    def upload_file(self, local_path: str, remote_filename: str) -> Dict[str, Any]:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Source file not found: {local_path}")

        dest_path = os.path.join(self.base_directory, remote_filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        source_sha = compute_sha256(local_path)
        source_size = os.path.getsize(local_path)

        shutil.copy2(local_path, dest_path)

        dest_sha = compute_sha256(dest_path)
        dest_size = os.path.getsize(dest_path)

        if source_sha != dest_sha or source_size != dest_size:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise IOError(
                f"BACKUP_UPLOAD_CORRUPTION: SHA-256 or size mismatch during local upload of {remote_filename}."
            )

        receipt = {
            "remote_filename": remote_filename,
            "destination": dest_path,
            "size_bytes": dest_size,
            "sha256": dest_sha,
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "is_external": False,
        }
        logger.info(f"Stored backup file locally: {remote_filename} ({dest_size} bytes)")
        return receipt

    def download_file(self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None) -> str:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: download_file {remote_filename}")

        src_path = os.path.join(self.base_directory, remote_filename)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Backup file not found in storage: {remote_filename}")

        os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)

        with open(src_path, "rb") as f_in, open(local_dest_path, "wb") as f_out:
            while chunk := f_in.read(65536):
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    if os.path.exists(local_dest_path):
                        os.remove(local_dest_path)
                    raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED_DURING_STREAM: download_file {remote_filename}")
                f_out.write(chunk)

        src_sha = compute_sha256(src_path)
        dst_sha = compute_sha256(local_dest_path)
        if src_sha != dst_sha:
            if os.path.exists(local_dest_path):
                os.remove(local_dest_path)
            raise IOError(f"DOWNLOAD_CORRUPTION: SHA-256 mismatch downloading {remote_filename}")

        return local_dest_path

    def list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("STORAGE_DEADLINE_EXCEEDED: list_backups")

        results = []
        if not os.path.exists(self.base_directory):
            return results

        for root, _, files in os.walk(self.base_directory):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("STORAGE_DEADLINE_EXCEEDED: list_backups")
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.base_directory)
                results.append({
                    "remote_filename": rel_path.replace("\\", "/"),
                    "size_bytes": os.path.getsize(full_path),
                    "modified_at": datetime.utcfromtimestamp(os.path.getmtime(full_path)).isoformat() + "Z",
                })
        results.sort(key=lambda x: x["modified_at"], reverse=True)
        return results

    def delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: delete_backup {remote_filename}")

        target = os.path.join(self.base_directory, remote_filename)
        if os.path.exists(target):
            os.remove(target)
            logger.info(f"Deleted backup file: {remote_filename}")
            return True
        return False

    def get_storage_destination(self) -> str:
        return f"file://{self.base_directory}"

    def is_external(self) -> bool:
        return False

    def get_backup_metadata(self, remote_filename: str) -> Optional[Dict[str, Any]]:
        target = os.path.join(self.base_directory, remote_filename)
        if not os.path.exists(target):
            return None
        if remote_filename in self._custom_metadata:
            meta = dict(self._custom_metadata[remote_filename])
            if "size_bytes" not in meta:
                meta["size_bytes"] = os.path.getsize(target)
            if "destination" not in meta:
                meta["destination"] = target
            if "remote_filename" not in meta:
                meta["remote_filename"] = remote_filename
            return meta
        return {
            "remote_filename": remote_filename,
            "size_bytes": os.path.getsize(target),
            "modified_at": datetime.utcfromtimestamp(os.path.getmtime(target)).isoformat() + "Z",
            "sha256": compute_sha256(target),
            "destination": target,
        }


class S3CompatibleStorageAdapter(BackupStorageAdapter):
    """
    Stores backups in an external S3-compatible object storage bucket
    (AWS S3, Cloudflare R2, Google Cloud Storage, or MinIO).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "smart_money_backups",
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: str = "us-west-2",
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self._s3_client = None

    def _get_client(self, deadline_monotonic: Optional[float] = None, max_attempts: int = 1):
        timeout = 30.0
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("STORAGE_DEADLINE_EXCEEDED: client init")
            # Bound socket connect and read timeouts by remaining cycle budget
            timeout = max(0.2, min(remaining, 10.0))

        # If a cached client exists and no specific deadline was requested, reuse it
        if deadline_monotonic is None and self._s3_client is not None:
            return self._s3_client

        try:
            import boto3
            from botocore.config import Config
            config = Config(
                connect_timeout=timeout,
                read_timeout=timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"},
            )
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region_name,
                config=config,
            )
            if deadline_monotonic is None:
                self._s3_client = client
            return client
        except ImportError:
            raise RuntimeError("boto3 is required for S3CompatibleStorageAdapter.")

    def _key(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def upload_file(self, local_path: str, remote_filename: str) -> Dict[str, Any]:
        s3 = self._get_client()
        key = self._key(remote_filename)
        source_sha = compute_sha256(local_path)
        source_size = os.path.getsize(local_path)

        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=f,
                Metadata={"sha256": source_sha},
            )

        # Gate 2: Verify upload via head_object
        head = s3.head_object(Bucket=self.bucket, Key=key)
        uploaded_size = head.get("ContentLength", 0)
        if uploaded_size != source_size:
            raise IOError(f"R2_UPLOAD_SIZE_MISMATCH: Uploaded size {uploaded_size} != source size {source_size}")

        # Gate 2: Verify metadata SHA-256 header (do not assume ETag equals SHA-256)
        meta = head.get("Metadata", {})
        meta_sha = meta.get("sha256")
        if not meta_sha:
            raise IOError(f"R2_UPLOAD_METADATA_MISSING: Metadata 'sha256' header not returned by S3/R2 for {remote_filename}")
        if meta_sha.lower() != source_sha.lower():
            raise IOError(
                f"R2_UPLOAD_METADATA_CORRUPTION: Remote metadata sha256 '{meta_sha}' != local source '{source_sha}'"
            )

        # Gate 2: End-to-end content checksum verification via streaming readback
        # In S3/R2, ETag is MD5 or multipart MD5 (e.g. hash-N), never true SHA-256.
        # We perform an independent streaming content readback to verify true byte digest.
        hasher = hashlib.sha256()
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        body = obj.get("Body")
        if body:
            while chunk := body.read(65536):
                hasher.update(chunk)
        remote_stream_sha = hasher.hexdigest()

        if remote_stream_sha.lower() != source_sha.lower():
            raise IOError(
                f"R2_UPLOAD_CONTENT_CORRUPTION: Remote content SHA-256 '{remote_stream_sha}' != local source '{source_sha}'"
            )

        etag_str = (head.get("ETag") or "").strip('"')
        is_multipart = bool("-" in etag_str)

        receipt = {
            "remote_filename": remote_filename,
            "destination": f"s3://{self.bucket}/{key}",
            "size_bytes": uploaded_size,
            "sha256": source_sha,
            "verified_remote_sha256": remote_stream_sha,
            "verified_metadata_sha256": meta_sha,
            "etag": etag_str,
            "is_multipart": is_multipart,
            "integrity_verified": True,
            "verification_method": "END_TO_END_STREAMING_SHA256",
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "is_external": True,
        }
        logger.info(f"Uploaded and verified backup on S3/R2: s3://{self.bucket}/{key} (sha256: {source_sha[:8]}...)")
        return receipt

    def download_file(self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None) -> str:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: download_file {remote_filename}")

        s3 = self._get_client(deadline_monotonic=deadline_monotonic, max_attempts=1)
        key = self._key(remote_filename)
        os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)

        try:
            obj = s3.get_object(Bucket=self.bucket, Key=key)
            body = obj.get("Body")
            if not body:
                raise IOError(f"EMPTY_RESPONSE_BODY: {remote_filename}")

            with open(local_dest_path, "wb") as f:
                while True:
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        body.close()
                        if os.path.exists(local_dest_path):
                            os.remove(local_dest_path)
                        raise TimeoutError(
                            f"STORAGE_DEADLINE_EXCEEDED_DURING_STREAM: download_file {remote_filename}"
                        )
                    chunk = body.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(local_dest_path):
                os.remove(local_dest_path)
            if isinstance(e, TimeoutError):
                raise
            raise IOError(f"DOWNLOAD_FAILED: Could not download {remote_filename} ({e})") from e

        # Post-download verification
        try:
            head = s3.head_object(Bucket=self.bucket, Key=key)
            expected_sha = head.get("Metadata", {}).get("sha256")
            if expected_sha:
                actual_sha = compute_sha256(local_dest_path)
                if actual_sha.lower() != expected_sha.lower():
                    if os.path.exists(local_dest_path):
                        os.remove(local_dest_path)
                    raise IOError(
                        f"DOWNLOAD_CORRUPTION: SHA-256 mismatch downloading {remote_filename} "
                        f"(expected {expected_sha}, got {actual_sha})"
                    )
        except Exception as e:
            if isinstance(e, IOError):
                raise
            logger.debug(f"Could not verify metadata SHA-256 for {remote_filename}: {e}")

        return local_dest_path

    def list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("STORAGE_DEADLINE_EXCEEDED: list_backups")

        s3 = self._get_client(deadline_monotonic=deadline_monotonic, max_attempts=1)
        paginator = s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("STORAGE_DEADLINE_EXCEEDED_PAGINATING: list_backups")
            for item in page.get("Contents", []):
                rel_name = item["Key"][len(self.prefix) + 1 :] if self.prefix else item["Key"]
                results.append({
                    "remote_filename": rel_name,
                    "size_bytes": item["Size"],
                    "modified_at": item["LastModified"].isoformat(),
                })
        return results

    def delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: delete_backup {remote_filename}")

        s3 = self._get_client(deadline_monotonic=deadline_monotonic, max_attempts=1)
        key = self._key(remote_filename)
        try:
            s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete remote backup {remote_filename}: {e}")
            return False

    def get_storage_destination(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def is_external(self) -> bool:
        return True

    def get_backup_metadata(self, remote_filename: str) -> Optional[Dict[str, Any]]:
        s3 = self._get_client()
        key = self._key(remote_filename)
        try:
            head = s3.head_object(Bucket=self.bucket, Key=key)
            return {
                "remote_filename": remote_filename,
                "size_bytes": head.get("ContentLength", 0),
                "modified_at": head.get("LastModified").isoformat() if head.get("LastModified") else None,
                "sha256": head.get("Metadata", {}).get("sha256"),
                "destination": f"s3://{self.bucket}/{key}",
            }
        except Exception:
            return None


def get_backup_storage_adapter(override_dir: Optional[str] = None) -> BackupStorageAdapter:
    """
    Factory resolving the appropriate storage adapter.
    Supports Cloudflare R2 (ARX_BACKUP_PROVIDER=cloudflare_r2 or ARX_R2_* variables)
    as well as generic S3-compatible providers.
    Enforces FAIL-CLOSED semantics when external custody is configured or mandatory:
    silent fallback to local storage is strictly prohibited.
    """
    provider = os.environ.get("ARX_BACKUP_PROVIDER", "").strip().lower()
    external_mandatory = os.environ.get("ARX_BACKUP_EXTERNAL_MANDATORY", "").strip().lower() in ("true", "1", "yes")

    # Bucket resolution
    bucket = (
        os.environ.get("ARX_BACKUP_BUCKET")
        or os.environ.get("SMART_MONEY_BACKUP_S3_BUCKET")
    )

    # Prefix resolution (default to 'rehearsal' for R2 rehearsal, 'smart_money_backups' for S3)
    prefix = (
        os.environ.get("ARX_BACKUP_PREFIX")
        or os.environ.get("SMART_MONEY_BACKUP_S3_PREFIX")
        or ("rehearsal" if provider == "cloudflare_r2" else "smart_money_backups")
    )

    # Cloudflare R2 specific resolution
    r2_account_id = os.environ.get("ARX_R2_ACCOUNT_ID")
    endpoint = (
        os.environ.get("ARX_R2_ENDPOINT")
        or (f"https://{r2_account_id}.r2.cloudflarestorage.com" if r2_account_id else None)
        or os.environ.get("SMART_MONEY_BACKUP_S3_ENDPOINT")
    )
    access_key = (
        os.environ.get("ARX_R2_ACCESS_KEY_ID")
        or os.environ.get("SMART_MONEY_BACKUP_S3_ACCESS_KEY")
    )
    secret_key = (
        os.environ.get("ARX_R2_SECRET_ACCESS_KEY")
        or os.environ.get("SMART_MONEY_BACKUP_S3_SECRET_KEY")
    )

    is_r2 = bool(provider == "cloudflare_r2" or r2_account_id or (endpoint and "r2.cloudflarestorage.com" in endpoint))
    provider_name = "cloudflare_r2" if is_r2 else "s3"
    region = "auto" if is_r2 else os.environ.get("SMART_MONEY_BACKUP_S3_REGION", "us-west-2")

    # If external provider is requested or external custody is mandatory: FAIL-CLOSED
    if provider in ("cloudflare_r2", "s3") or external_mandatory:
        missing = []
        if not bucket:
            missing.append("bucket (ARX_BACKUP_BUCKET / SMART_MONEY_BACKUP_S3_BUCKET)")
        if not endpoint:
            missing.append("endpoint (ARX_R2_ENDPOINT / ARX_R2_ACCOUNT_ID / SMART_MONEY_BACKUP_S3_ENDPOINT)")
        if not access_key:
            missing.append("access_key (ARX_R2_ACCESS_KEY_ID / SMART_MONEY_BACKUP_S3_ACCESS_KEY)")
        if not secret_key:
            missing.append("secret_key (ARX_R2_SECRET_ACCESS_KEY / SMART_MONEY_BACKUP_S3_SECRET_KEY)")

        if missing:
            raise ValueError(
                f"EXTERNAL_BACKUP_CONFIG_ERROR: Missing required {provider_name} configuration: {', '.join(missing)}. "
                f"Fail-closed: silent fallback to local storage is strictly prohibited."
            )

        try:
            adapter = S3CompatibleStorageAdapter(
                bucket=bucket,
                prefix=prefix,
                endpoint_url=endpoint,
                access_key_id=access_key,
                secret_access_key=secret_key,
                region_name=region,
            )
            adapter.provider_name = provider_name
            return adapter
        except Exception as e:
            raise RuntimeError(
                f"EXTERNAL_BACKUP_INIT_ERROR: Failed to initialize {provider_name} adapter ({e}). "
                f"Fail-closed: silent fallback to local storage is strictly prohibited."
            ) from e

    # Only when provider is explicitly 'local' or empty and external is not mandatory
    if provider and provider != "local":
        raise ValueError(f"UNSUPPORTED_BACKUP_PROVIDER: Unknown backup provider '{provider}'")

    base_dir = override_dir or os.environ.get(
        "SMART_MONEY_BACKUP_DIR",
        os.path.join(os.path.expanduser("~"), ".arx_backups", "smart_money")
    )
    adapter = LocalOffVolumeStorageAdapter(base_directory=base_dir)
    adapter.provider_name = "local_off_volume"
    adapter.bucket = "local"
    adapter.prefix = ""
    return adapter
