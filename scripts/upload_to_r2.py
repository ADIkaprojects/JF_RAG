"""
scripts/upload_to_r2.py
━━━━━━━━━━━━━━━━━━━━━━━
One-time script: uploads all runtime-necessary files to Cloudflare R2.
Run this once locally before deploying to production.

Usage:
    python scripts/upload_to_r2.py

Requirements:
    - .env file must exist with R2_* credentials set
    - Files listed in UPLOAD_MANIFEST should exist locally
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.file_resolver import upload_local_file  # noqa: E402


def to_r2_key(path: str) -> str:
    """Normalise a local path to an R2 object key."""
    return str(path).replace("\\", "/").lstrip("./")


# ── Files to upload (Bucket A) ────────────────────────────────────────────────
# Edit this list if you add more runtime-necessary artifacts.
# Paths are relative to the project root and should match CLOUD_FILES.md.
UPLOAD_MANIFEST = [
    # Text FAISS indexes
    "vectorstore/faiss/index.bin",
    "vectorstore/faiss/chunks.json",

    # BM25 index
    "vectorstore/bm25.pkl",

    # Image FAISS indexes
    "vectorstore/image_faiss/index.bin",
    "vectorstore/image_faiss/chunks.json",

    # Optional parent index (if generated)
    "vectorstore/parent_index.json",

    # Source documents: upload all files under these folders as needed.
    # You can either keep these as directories and upload recursively
    # via another script, or expand them into concrete file paths.
    # Here we include the top-level directories to make intent explicit.
    # "data/sample_document/...",
    # "data/sample_image/...",
    # "data/csv/...",
    # "data/dataset/...",
    # "data/text/...",
]


# ── Production Optimization ──────────────────────────────────────────────────
# REMOVED: Original source data directories (data/sample_document, data/csv, etc.)
# These are NOT needed in production since all content has been:
#   - Extracted and chunked into FAISS text index vectorstore/faiss/
#   - Embedded as vectors in vectorstore/faiss/index.bin
#   - Images extracted to processed/images/ and embedded in vectorstore/image_faiss/
#
# Source files are ONLY used during the indexing phase (orchestrate.py), which has
# already completed. Do NOT upload them to R2 to save bandwidth and storage costs.
#
# NOTE: For processed/images/ folder, either:
#   A) Uncomment the line below and re-run this script, OR
#   B) Manually upload processed/images/ folder to R2 via dashboard
# Then ensure api/main.py is updated to serve images from R2 in production mode.

# OPTIONAL: Uncomment below to add extracted images to upload
for file in (PROJECT_ROOT / "processed" / "images").rglob("*"):
    if file.is_file():
        UPLOAD_MANIFEST.append(to_r2_key(file.relative_to(PROJECT_ROOT)))


def main() -> None:
    bucket = os.getenv("R2_BUCKET_NAME", "")
    print(f"Uploading {len(UPLOAD_MANIFEST)} files to R2 bucket: {bucket or '(unset)'}")
    print()

    success: list[str] = []
    failed: list[str] = []

    for file_path in UPLOAD_MANIFEST:
        local = PROJECT_ROOT / file_path
        if not local.exists():
            print(f"  SKIP (not found locally): {file_path}")
            failed.append(file_path)
            continue

        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"  Uploading: {file_path} ({size_mb:.1f} MB) ...", end=" ")

        try:
            # Use the relative path as the R2 key so it matches what
            # IndexLoader and ImageIndexLoader expect when resolving.
            upload_local_file(str(local), file_path)
            print("OK")
            success.append(file_path)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")
            failed.append(file_path)

    print()
    print(f"Done. {len(success)} uploaded, {len(failed)} failed/skipped.")
    if failed:
        print("Failed/skipped entries:")
        for fp in failed:
            print(f"  - {fp}")


if __name__ == "__main__":
    main()

