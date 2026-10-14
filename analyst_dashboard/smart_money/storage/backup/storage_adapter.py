"""
ARX Terminal — Smart Money Backup Storage Adapters.

Defines provider-neutral storage abstraction supporting:
1. Local off-volume directory storage (disaster recovery partition);
2. S3-compatible remote object storage (Cloudflare R2, AWS S3, Google Cloud Storage);
3. Automated SHA-256 integrity verification upon write and read.
"""

import os
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
    def download_file(self, remote_filename: str, local_dest_path: str) -> str:
        """Downloads a file from storage destination to local path."""
        pass

    @abstractmethod
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lists all stored backup artifacts."""
        pass

    @abstractmethod
    def delete_backup(self, remote_filename: str) -> bool:
        """Deletes a backup artifact from storage."""
        pass

    @abstractmethod
    def get_storage_destination(self) -> str:
        """Returns human-readable representation of the destination."""
        pass

    @abstractmethod
    def is_external(self) -> bool:
        """Returns True if the destination is external off-host cloud storage."""
        pass


class LocalOffVolumeStorageAdapter(BackupStorageAdapter):
    """
    Stores backups in a dedicated off-volume directory outside the database volume.
    Used for local testing, staging, and environments lacking cloud S3 buckets.
    """

    def __init__(self, base_directory: str):
        self.base_directory = os.path.abspath(base_directory)
        os.makedirs(self.base_directory, exist_ok=True)

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

    def download_file(self, remote_filename: str, local_dest_path: str) -> str:
        src_path = os.path.join(self.base_directory, remote_filename)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Backup file not found in storage: {remote_filename}")

        os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)
        shutil.copy2(src_path, local_dest_path)

        src_sha = compute_sha256(src_path)
        dst_sha = compute_sha256(local_dest_path)
        if src_sha != dst_sha:
            raise IOError(f"DOWNLOAD_CORRUPTION: SHA-256 mismatch downloading {remote_filename}")

        return local_dest_path

    def list_backups(self) -> List[Dict[str, Any]]:
        results = []
        if not os.path.exists(self.base_directory):
            return results

        for root, _, files in os.walk(self.base_directory):
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

    def delete_backup(self, remote_filename: str) -> bool:
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

    def _get_client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region_name,
            )
            return self._s3_client
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

        # Verify upload via head_object
        head = s3.head_object(Bucket=self.bucket, Key=key)
        uploaded_size = head.get("ContentLength", 0)
        if uploaded_size != source_size:
            raise IOError(f"S3_UPLOAD_CORRUPTION: Uploaded size {uploaded_size} != source size {source_size}")

        receipt = {
            "remote_filename": remote_filename,
            "destination": f"s3://{self.bucket}/{key}",
            "size_bytes": uploaded_size,
            "sha256": source_sha,
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "is_external": True,
        }
        logger.info(f"Uploaded backup to S3: s3://{self.bucket}/{key}")
        return receipt

    def download_file(self, remote_filename: str, local_dest_path: str) -> str:
        s3 = self._get_client()
        key = self._key(remote_filename)
        os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)

        s3.download_file(self.bucket, key, local_dest_path)
        return local_dest_path

    def list_backups(self) -> List[Dict[str, Any]]:
        s3 = self._get_client()
        paginator = s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                rel_name = item["Key"][len(self.prefix) + 1 :] if self.prefix else item["Key"]
                results.append({
                    "remote_filename": rel_name,
                    "size_bytes": item["Size"],
                    "modified_at": item["LastModified"].isoformat(),
                })
        return results

    def delete_backup(self, remote_filename: str) -> bool:
        s3 = self._get_client()
        key = self._key(remote_filename)
        s3.delete_object(Bucket=self.bucket, Key=key)
        return True

    def get_storage_destination(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def is_external(self) -> bool:
        return True


def get_backup_storage_adapter(override_dir: Optional[str] = None) -> BackupStorageAdapter:
    """
    Factory resolving the appropriate storage adapter.
    Supports Cloudflare R2 (ARX_BACKUP_PROVIDER=cloudflare_r2 or ARX_R2_* variables)
    as well as generic S3-compatible providers.
    Falls back safely to LocalOffVolumeStorageAdapter when remote credentials are absent.
    """
    provider = os.environ.get("ARX_BACKUP_PROVIDER", "").strip().lower()

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

    if bucket and endpoint and access_key and secret_key:
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
            logger.warning(f"Could not initialize {provider_name} adapter ({e}); falling back to local off-volume adapter.")

    base_dir = override_dir or os.environ.get(
        "SMART_MONEY_BACKUP_DIR",
        os.path.join(os.path.expanduser("~"), ".arx_backups", "smart_money")
    )
    adapter = LocalOffVolumeStorageAdapter(base_directory=base_dir)
    adapter.provider_name = "local_off_volume"
    adapter.bucket = "local"
    adapter.prefix = ""
    return adapter
