"""
modal_app.py — Modal deployment for JF_RAG FastAPI backend
===========================================================
Deploy (production):  modal deploy modal_app.py
Serve  (dev / hot-reload):  modal serve modal_app.py

Before deploying:
  1. modal setup          (authenticate once)
  2. modal secret create jf-rag-secrets \\
       GROQ_API_KEYS=... \\
       HF_API_TOKEN=... \\
       R2_ACCOUNT_ID=... \\
       R2_ACCESS_KEY_ID=... \\
       R2_SECRET_ACCESS_KEY=... \\
       R2_BUCKET_NAME=jf-rag-data \\
       ENVIRONMENT=production \\
       EMBEDDING_MODEL=intfloat/e5-large-v2 \\
       CLIP_MODEL=openai/clip-vit-base-patch32 \\
       RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \\
       CORS_ORIGINS=https://jf-rag.vercel.app,http://localhost:8080
  3. modal volume create jf-rag-model-cache   (first time only)
"""

import modal

# ── App ───────────────────────────────────────────────────────────────────────
app = modal.App("jf-rag-backend")

# ── Persistent volume: cache HuggingFace model weights ───────────────────────
# e5-large-v2 (~1.3 GB) + CLIP (~600 MB) + cross-encoder (~90 MB)
# Without this volume they re-download on every cold start.
model_cache  = modal.Volume.from_name("jf-rag-model-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libraries
    .apt_install(
        "tesseract-ocr",    # OCR for scanned PDFs
        "libgl1",           # OpenCV headless dependency
        "libglib2.0-0",     # OpenCV headless dependency
        "cmake",            # Required to compile dlib (face_recognition)
        "build-essential",  # C++ compiler for dlib
        "libopenblas-dev",  # BLAS — speeds up FAISS + numpy
    )
    # Copy project source (skip large generated/binary directories)
    .copy_local_dir(
        ".",
        "/app",
        ignore=[
            ".git",
            "venv", ".venv", "__pycache__", "*.pyc",
            "vectorstore",          # indexes fetched from R2 at runtime
            "processed",            # pipeline output — not needed at serve time
            "cache",                # diskcache — recreated in the container
            "data",                 # source documents — not needed at serve time
            "known_faces",          # binary images
            "unredacted/processed_photos",
            "frontend/node_modules",
            "frontend/dist",
            "frontend/.env",
            "*.egg-info",
            "node_modules",
        ],
    )
    # Install all Python dependencies
    .pip_install_from_requirements("/app/requirements.txt")
    # Bake the spaCy model into the image so it's available on cold start
    .run_commands("python -m spacy download en_core_web_trf")
    .workdir("/app")
)


# ── FastAPI ASGI endpoint ─────────────────────────────────────────────────────
@app.function(
    image=image,
    # Modal secret — create via: modal secret create jf-rag-secrets KEY=VALUE ...
    secrets=[modal.Secret.from_name("jf-rag-secrets")],
    # Reuse model weights volume across restarts
    volumes={HF_CACHE_DIR: model_cache},
    # Sleep after 5 minutes of no traffic (saves credits, cold start ~30-60 s)
    container_idle_timeout=300,
    # Max wall-clock time per request (FAISS + reranker + LLM can be slow)
    timeout=300,
    # Memory: e5-large ~1.3 GB + CLIP ~600 MB + reranker ~90 MB + FAISS + headroom
    memory=8192,
    cpu=2.0,
)
@modal.asgi_app()
def fastapi_app():
    from api.main import app as _app
    return _app
