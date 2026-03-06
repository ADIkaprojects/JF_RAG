"""
lib/file_resolver.py — Feature: Cloud File Resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Environment-aware file loader.

  Development (ENV != 'production'):
    → Reads directly from local filesystem (fast, no network)

  Production (ENV == 'production') OR file not found locally:
    → Fetches from Cloudflare R2 bucket via S3-compatible API

Dependencies:
    pip install boto3

Environment variables required in .env:
    ENVIRONMENT=development          # or 'production'
    R2_ACCOUNT_ID=<cloudflare id>
    R2_ACCESS_KEY_ID=<r2 token key>
    R2_SECRET_ACCESS_KEY=<r2 token secret>
    R2_BUCKET_NAME=<your bucket name>
"""

import os
import io
import logging
from pathlib import Path
from typing import Union

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ── Environment detection ─────────────────────────────────────────────────────

ENVIRONMENT   = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

# ── R2 client (lazy-initialised) ──────────────────────────────────────────────

_r2_client = None

def _get_r2_client():
    """Return a cached boto3 S3 client configured for Cloudflare R2."""
    global _r2_client
    if _r2_client is not None:
        return _r2_client

    account_id       = os.getenv("R2_ACCOUNT_ID")
    access_key_id    = os.getenv("R2_ACCESS_KEY_ID")
    secret_key       = os.getenv("R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key_id, secret_key]):
        raise EnvironmentError(
            "R2 credentials are not set. "
            "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY in your .env file."
        )

    _r2_client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_key,
    )
    return _r2_client


def _get_bucket() -> str:
    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        raise EnvironmentError("R2_BUCKET_NAME is not set in your .env file.")
    return bucket


# ── Core functions ────────────────────────────────────────────────────────────

def resolve_file(relative_path: str) -> bytes:
    """
    Load a file as bytes — from local disk in development, from R2 in production.

    Args:
        relative_path: Path relative to project root, e.g. 'data/docs.pdf'
                       or 'faiss_index/index.faiss'

    Returns:
        bytes — file content as a bytes object

    Usage:
        content = resolve_file("data/my_document.pdf")
        # Pass to pdf parser:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
    """
    local_path = Path.cwd() / relative_path

    if not IS_PRODUCTION and local_path.exists():
        logger.info(f"[LOCAL] Reading: {relative_path}")
        return local_path.read_bytes()

    logger.info(f"[R2] Fetching: {relative_path}")
    try:
        response = _get_r2_client().get_object(
            Bucket=_get_bucket(),
            Key=relative_path,
        )
        return response["Body"].read()
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        raise FileNotFoundError(
            f"File not found in R2: '{relative_path}' (R2 error: {error_code})"
        ) from e


def resolve_file_stream(relative_path: str) -> io.BytesIO:
    """
    Same as resolve_file() but returns a BytesIO stream instead of raw bytes.
    Use this when the consumer expects a file-like object (e.g. FAISS, PyMuPDF).

    Usage:
        stream = resolve_file_stream("faiss_index/index.faiss")
        index = faiss.read_index_binary(stream)  # if supported
    """
    return io.BytesIO(resolve_file(relative_path))


def list_files(prefix: str = "") -> list[str]:
    """
    List files — from local directory in development, from R2 prefix in production.

    Args:
        prefix: Directory path relative to project root, e.g. 'data/'

    Returns:
        List of relative file paths (strings)

    Usage:
        files = list_files("data/")
        # Returns ['data/doc1.pdf', 'data/doc2.csv', ...]
    """
    local_dir = Path.cwd() / prefix

    if not IS_PRODUCTION and local_dir.exists() and local_dir.is_dir():
        logger.info(f"[LOCAL] Listing: {prefix}")
        return [
            str(Path(prefix) / f.name)
            for f in local_dir.iterdir()
            if f.is_file()
        ]

    logger.info(f"[R2] Listing prefix: {prefix}")
    try:
        paginator = _get_r2_client().get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=_get_bucket(), Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
    except ClientError as e:
        logger.error(f"Failed to list R2 prefix '{prefix}': {e}")
        return []


def upload_file(relative_path: str, content: bytes) -> None:
    """
    Upload a file to R2. Used for syncing locally generated index files
    to cloud storage after ingestion.

    Args:
        relative_path: R2 key / destination path, e.g. 'faiss_index/index.faiss'
        content: File content as bytes

    Usage:
        with open("faiss_index/index.faiss", "rb") as f:
            upload_file("faiss_index/index.faiss", f.read())
    """
    _get_r2_client().put_object(
        Bucket=_get_bucket(),
        Key=relative_path,
        Body=content,
    )
    logger.info(f"[R2] Uploaded: {relative_path}")


def upload_local_file(local_path: str, r2_key: str = None) -> None:
    """
    Convenience wrapper — reads a local file and uploads it to R2.

    Args:
        local_path: Absolute or relative path to local file
        r2_key: R2 destination key. Defaults to local_path if not provided.
    """
    path = Path(local_path)
    key  = r2_key or local_path
    upload_file(key, path.read_bytes())

