"""
modal_app.py — JF_RAG Backend on Modal
========================================
Deploy with:
    modal deploy modal_app.py

Check logs with:
    modal app logs jf-rag-backend
"""

import modal
import os

# ── Container image ──────────────────────────────────────────────────────────
# Build once, cached — only rebuilds when requirements.txt changes.
_IGNORE = [
    ".git", "venv", ".venv", "__pycache__", "*.pyc",
    "vectorstore", "processed", "cache", "data/sample_document",
    "data/sample_image", "data/csv", "known_faces",
    "unredacted/processed_photos", "frontend/node_modules",
    "frontend/dist", "frontend/.env", "node_modules",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "cmake",
        "build-essential",
    )
    # Install deps first (no local files needed yet)
    .pip_install_from_requirements("requirements.txt")
    # Pre-bake heavy models so cold starts are fast (~30 s instead of ~3 min)
    .run_commands(
        'python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'intfloat/e5-large-v2\')"',
        'python -c "from sentence_transformers import CrossEncoder; CrossEncoder(\'cross-encoder/ms-marco-MiniLM-L-6-v2\')"',
        'python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained(\'openai/clip-vit-base-patch32\'); CLIPProcessor.from_pretrained(\'openai/clip-vit-base-patch32\')"',
    )
    # add_local_dir MUST come last; copy=True bakes files into the image layer
    .add_local_dir(".", remote_path="/app", ignore=_IGNORE, copy=True)
    .workdir("/app")
)

# ── Modal app ────────────────────────────────────────────────────────────────
app = modal.App("jf-rag-backend", image=image)

# ── Secrets — set these in Modal dashboard under "Secrets" ──────────────────
# modal secret create jf-rag-secrets \
#   ENVIRONMENT=production \
#   GROQ_API_KEYS=key1,key2 \
#   R2_ACCOUNT_ID=xxx \
#   R2_ACCESS_KEY_ID=xxx \
#   R2_SECRET_ACCESS_KEY=xxx \
#   R2_BUCKET_NAME=xxx \
#   EMBEDDING_MODEL=intfloat/e5-large-v2 \
#   CORS_ORIGINS=https://your-vercel-app.vercel.app
secrets = [modal.Secret.from_name("jf-rag-secrets")]


# ── FastAPI ASGI app ─────────────────────────────────────────────────────────
@app.function(
    memory=4096,
    cpu=2.0,
    scaledown_window=300,
    timeout=600,
    secrets=secrets,
)
@modal.asgi_app()
def fastapi_app():
    from api.main import app as _app
    return _app