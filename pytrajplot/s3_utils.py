"""S3 utility functions for pytrajplot AWS integration."""
import logging
import os
from typing import Any

from botocore.exceptions import ClientError

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def download_s3_prefix(s3_client: Any, bucket: str, prefix: str, local_dir: str) -> None:
    """Download all objects under an S3 prefix into a local directory, preserving relative paths.

    Raises:
        RuntimeError: If no files are found under the given S3 prefix.
        RuntimeError: If the bucket does not exist or access is denied.
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                relative_path = key[len(prefix):].lstrip("/")
                if not relative_path:
                    continue
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
                s3_client.download_file(bucket, key, local_path)
                downloaded += 1
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchBucket":
            raise RuntimeError(f"S3 bucket does not exist: {bucket}") from e
        if code == "AccessDenied":
            raise RuntimeError(f"Access denied reading from s3://{bucket}/{prefix}") from e
        raise

    if downloaded == 0:
        raise RuntimeError(f"No files found at s3://{bucket}/{prefix} — bucket prefix is empty or does not exist.")


def upload_dir_to_s3(s3_client: Any, local_dir: str, bucket: str, prefix: str) -> None:
    """Upload all files in a local directory to an S3 prefix.

    Raises:
        RuntimeError: If the output directory is empty (pytrajplot produced no output).
        RuntimeError: If the bucket does not exist or access is denied.
    """
    uploaded = 0
    try:
        for root, _, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{prefix.rstrip('/')}/{relative_path}" if prefix else relative_path
                logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, s3_key)
                s3_client.upload_file(local_path, bucket, s3_key)
                uploaded += 1
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchBucket":
            raise RuntimeError(f"S3 bucket does not exist: {bucket}") from e
        if code == "AccessDenied":
            raise RuntimeError(f"Access denied writing to s3://{bucket}/{prefix}") from e
        raise

    if uploaded == 0:
        raise RuntimeError(f"No output files were produced — nothing uploaded to s3://{bucket}/{prefix}.")
