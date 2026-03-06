"""
MultiModal RAG API – FastAPI Application
=========================================
Upgraded to incorporate the Advanced RAG notebook's patterns (Cell 28):
  - startup_event registers parent chunks for parent-child retrieval (notebook Cell 28)
  - /query uses retrieve_with_parent_child when use_parent_child=True (notebook Cell 18)
  - /status endpoint mirrors notebook ProductionRAGSystem.get_system_status() (Cell 28)
  - Performance metrics tracking added (notebook Cell 28 self.performance_metrics)
  - All original endpoints (/query, /retrieve, /cache/*) and guardrails unchanged

Guardrails: KEPT as-is (user's existing implementation, not overwritten).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import io
import logging
import traceback
import diskcache
import hashlib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import asyncio

load_dotenv()

from llm.router import LLMRouter
from retrieval.retriever import HybridRetriever, ImageRetriever, ParentChunk, build_context
from retrieval.embeddings import IndexLoader, ImageIndexLoader, EmbeddingManager
from api.guardrails import guard_input
from config import EMBEDDING_MODEL, FAISS_DIR, BM25_PATH
from lib.file_resolver import resolve_file, upload_local_file, IS_PRODUCTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MultiModal RAG API",
    description="Retrieval-Augmented Generation for text, images, and CSV data",
    version="1.0.0",
)

# Enable CORS for React frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ── Feature 1: static image serving ─────────────────────────────────────────
# Serves processed/images/* as  GET /images/<filename>
# so the React frontend can load image thumbnails in ImageResultCards.
_IMAGES_DIR = os.path.join(os.getenv("OUTPUT_DIR", "./processed"), "images")
if not IS_PRODUCTION:
    # Development: serve images from local disk via StaticFiles
    os.makedirs(_IMAGES_DIR, exist_ok=True)
    app.mount("/images", StaticFiles(directory=_IMAGES_DIR), name="images")
# Production: images served via custom endpoint below (fetches from R2)

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
cache     = diskcache.Cache(CACHE_DIR)

retriever:        Optional[HybridRetriever]  = None
image_retriever:  Optional[ImageRetriever]   = None   # Gap Fix: image search
router:           Optional[LLMRouter]        = None
embedding_model:  Optional[EmbeddingManager] = None

# Notebook Cell 28 style: track performance metrics at the app level
# (replaces the in-class list in ProductionRAGSystem.performance_metrics)
_performance_metrics: List[Dict[str, Any]] = []


def to_r2_key(path: str | Path) -> str:
    """Normalise a local path to an R2 object key."""
    return str(path).replace("\\", "/").lstrip("./")


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:              str
    # Default retrieval settings (match UI: Top K=10, Rerank Top N=5)
    top_k:              int  = 10
    rerank_top_n:       int  = 5
    # Default provider: "auto" (UI: Auto recommended)
    force_provider:     Optional[str] = "auto"
    # Default LLM / retrieval toggles (match UI switches)
    use_cache:          bool = True       # Use Cache: ON
    use_hyde:           bool = True       # HyDE: ON
    use_query_rewrite:  bool = True
    # Parent-Child Retrieval: OFF by default
    use_parent_child:   bool = False
    # Include image retrieval results alongside text (Gap Fix #2)
    use_image_search:   bool = True


class RetrievalResult(BaseModel):
    chunk_id:    str
    text:        str
    source_file: str
    page:        int
    doc_type:    str
    # New: confidence score from notebook Cell 18
    confidence_score: float = 0.0
    # New: image-specific fields (populated for IMAGE results, empty for text)
    image_path:  Optional[str]  = None
    caption:     Optional[str]  = None
    geo_city:    Optional[str]  = None
    similarity:  Optional[float] = None


class QueryResponse(BaseModel):
    answer:              str
    provider:            str
    model:               str
    sources:             List[RetrievalResult]
    image_sources:       List[RetrievalResult] = []   # Gap Fix #2: image results
    error:               Optional[str] = None
    cached:              bool = False
    rewritten_query:     Optional[str] = None
    hyde_used:           bool = False
    guardrail_triggered: bool = False


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Initialise the RAG system (notebook Cell 28 ProductionRAGSystem.__init__ style).

    Steps mirror the notebook's component log:
      📄 Document Processing (embedding model)
      🗄️  Vector Storage     (FAISS index)
      🔍 Hybrid Retrieval    (BM25 + FAISS + reranker + LLMRouter)
    """
    global retriever, image_retriever, router, embedding_model

    logger.info("🚀 Initializing RAG system…")

    try:
        # ── Embedding model ──────────────────────────────────────────────
        logger.info(f"   📄 Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = EmbeddingManager(EMBEDDING_MODEL)

        # ── FAISS index ──────────────────────────────────────────────────
        faiss_dir = os.getenv("FAISS_TEXT_INDEX", FAISS_DIR)
        faiss_index, faiss_chunks = IndexLoader.load_faiss_index(faiss_dir)

        index_dim = faiss_index.d
        model_dim = embedding_model.embedding_dim
        if index_dim != model_dim:
            raise ValueError(
                f"Dimension mismatch: FAISS={index_dim}-dim, "
                f"model={model_dim}-dim. Re-run orchestrate.py --embed."
            )
        logger.info(f"   🗄️  FAISS OK: {faiss_index.ntotal} vectors, {index_dim}-dim")

        # ── BM25 index ───────────────────────────────────────────────────
        bm25_path = os.getenv("BM25_INDEX", BM25_PATH)
        if Path(bm25_path).exists():
            bm25_index, bm25_chunks = IndexLoader.load_bm25_index(bm25_path)
        else:
            logger.warning("⚠️  BM25 index not found — keyword search disabled")
            bm25_index, bm25_chunks = None, []

        # ── LLM Router ───────────────────────────────────────────────────
        router = LLMRouter(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

        # ── HybridRetriever (now includes parent-child + HyDE wiring) ───
        # The llm_client kwarg wires the router for generate_hyde_query /
        # generate_multi_queries inside retrieve_with_parent_child (notebook Cell 18)
        retriever = HybridRetriever(
            faiss_index=faiss_index,
            faiss_chunks=faiss_chunks,
            bm25_index=bm25_index,
            bm25_chunks=bm25_chunks,
            embedding_model=embedding_model,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            llm_client=router,
        )

        # ── (Optional) Load parent-child index if persisted ─────────────
        # When orchestrate.py --embed produces a parent_index.json alongside
        # the vectorstore, register ParentChunks so the retriever can expand
        # child hits back to their parent context (notebook Cell 18 pattern).
        parent_index_path = Path(os.getenv("PARENT_INDEX", "./vectorstore/parent_index.json"))
        if parent_index_path.exists():
            _load_parent_index(parent_index_path)
        else:
            logger.info("   ℹ️  No parent index found – parent-child retrieval will use self-context")

        # ── Image FAISS index (Gap Fix #1 & #2) ─────────────────────
        # Load the dedicated CLIP-space image index built by
        # orchestrate.py --images.  If it doesn't exist yet (e.g. first
        # run before --images has been executed) image search is silently
        # disabled — text retrieval continues normally.
        image_faiss_dir = Path(os.getenv(
            "IMAGE_FAISS_DIR", "./vectorstore/image_faiss"
        ))
        if image_faiss_dir.exists():
            try:
                img_index, img_chunks = ImageIndexLoader.load_image_faiss_index(
                    str(image_faiss_dir)
                )
                image_retriever = ImageRetriever(
                    image_faiss_index = img_index,
                    image_chunks      = img_chunks,
                    top_k             = 10,  # Default: can be overridden per-request
                )
                logger.info(
                    f"   🖼️  Image index: ✅  "
                    f"({img_index.ntotal} vectors, {len(img_chunks)} records)"
                )
            except Exception as e:
                logger.warning(f"   ⚠️  Image index failed to load: {e} — image search disabled")
                image_retriever = ImageRetriever()   # empty, search disabled
        else:
            logger.info(
                "   🖼️  Image index not found — run orchestrate.py --images first"
            )
            image_retriever = ImageRetriever()       # empty, search disabled

        logger.info("✅ RAG system initialised")
        logger.info("   📄 Embedding model: ✅")
        logger.info("   🗄️  Vector storage:  ✅")
        logger.info("   🛡️  Security layer:  ✅  (guardrails in api/guardrails.py)")
        logger.info("   🔍 Advanced retrieval: ✅  (HyDE + multi-query + parent-child)")
        logger.info("   🖼️  Image retrieval:  ✅  (CLIP cross-modal search)")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


def _load_parent_index(path: Path) -> None:
    """
    Register pre-built ParentChunks with the retriever at startup.
    Called once during startup if the file exists; safe to skip otherwise.

    In production, the parent index JSON may be loaded from Cloudflare R2 via file_resolver.
    """
    try:
        raw = resolve_file(to_r2_key(path))
        parent_data = json.loads(raw.decode("utf-8"))

        for entry in parent_data:
            pc = ParentChunk(
                parent_id       = entry["parent_id"],
                content         = entry["content"],
                document_id     = entry.get("document_id", ""),
                child_chunk_ids = entry.get("child_chunk_ids", []),
                start_char      = entry.get("start_char", 0),
                end_char        = entry.get("end_char", 0),
                page_numbers    = entry.get("page_numbers", []),
            )
            retriever.register_parent_chunk(pc)

        logger.info(f"✅ Loaded {len(parent_data)} parent chunks from {path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not load parent index: {e}")


def sync_indexes_to_r2() -> None:
    """Upload freshly built vector indexes to R2 after ingestion."""
    if not IS_PRODUCTION:
        # In development, indexes stay local — no need to upload
        return

    index_files = [
        # Text FAISS index
        "vectorstore/faiss/index.bin",
        "vectorstore/faiss/chunks.json",
        # BM25 index
        "vectorstore/bm25.pkl",
        # Image FAISS index
        "vectorstore/image_faiss/index.bin",
        "vectorstore/image_faiss/chunks.json",
        # Optional parent-child mapping index (if generated)
        "vectorstore/parent_index.json",
    ]

    for file_path in index_files:
        if Path(file_path).exists():
            upload_local_file(file_path)
            logger.info(f"Synced to R2: {file_path}")


def _get_classifier_client():
    """
    Return the fastest available LLM client for guardrail classification.
    Groq preferred (fast, cloud); falls back to Ollama (local).
    """
    if router.groq.available:
        return router.groq
    return router.ollama


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "features":  {
            "guardrails":       "Input-only, balanced (Layer1: hard rules + Layer2: LLM classifier)",
            "hyde":             True,
            "query_rewrite":    True,
            "hybrid_search":    True,
            "reranking":        True,
            "parent_child":     True,   # new
        },
    }


@app.get("/images/{image_path:path}")
async def serve_image(image_path: str):
    """
    Serve extracted images from either local disk (development) or R2 (production).
    
    In development: retrieves from ./processed/images/<image_path>
    In production:  retrieves from R2 bucket (jf-rag-data/processed/images/<image_path>)
    
    Frontend loads images via: /images/filename.jpg or /images/subfolder/filename.jpg
    """
    # Guard against path traversal attacks
    if ".." in image_path or image_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid image path")

    ext = image_path.lower().rsplit(".", 1)[-1] if "." in image_path else ""
    media_type = {
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")

    try:
        if IS_PRODUCTION:
            # Production: fetch image bytes from R2
            r2_key = f"processed/images/{image_path}"
            image_bytes = resolve_file(r2_key)
        else:
            # Development: StaticFiles mount handles most requests; this fallback
            # triggers only when the mount cannot find the file.
            local_path = Path(_IMAGES_DIR) / image_path
            if not local_path.exists():
                raise HTTPException(status_code=404, detail="Image not found")
            with open(local_path, "rb") as f:
                image_bytes = f.read()

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image retrieval failed for {image_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve image: {str(e)}")


@app.post("/query-with-file", response_model=QueryResponse)
async def query_with_file(
    query: str = Form(""),
    # Match UI defaults: Top K=10, Rerank Top N=5, Provider="auto"
    top_k: int = Form(10),
    rerank_top_n: int = Form(5),
    force_provider: str = Form("auto"),
    use_cache: bool = Form(True),
    use_hyde: bool = Form(True),
    # Parent-Child retrieval OFF by default
    use_parent_child: bool = Form(False),
    use_image_search: bool = Form(True),
    file: UploadFile = File(...),
):
    """
    Feature 2 — Query with an attached PDF or image file.

    The uploaded file is processed inline at query-time:
      • PDF  → text extracted via PyMuPDF (fitz), embedded with EmbeddingManager,
               searched against the text FAISS index, merged into retrieved_chunks.
      • PNG/JPEG/WEBP → CLIP embedding computed via pipelines.image_pipeline.get_clip_embedding(),
               searched against the image FAISS index via ImageRetriever.retrieve_by_vector(),
               merged into image_sources.

    Returns the same QueryResponse schema as /query so the frontend only needs
    to switch between api.query() and api.queryWithFile().
    """
    if retriever is None or router is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    # ── Read uploaded file bytes and enforce size/type limits ──────────────────
    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    filename = file.filename or ""
    content_type = file.content_type or ""

    allowed_exts = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"
    is_image = (
        content_type.startswith("image/")
        or any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp"))
    )

    if not (is_pdf or is_image) or not any(filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PDF, PNG, JPG, JPEG, WEBP.",
        )

    extra_text_chunks: list = []
    extra_image_sources: list = []

    # ── Guardrail on the text query (same as /query) ──────────────────────────
    classifier = _get_classifier_client()
    clean_query, guard_error = guard_input(query or "", classifier)
    if guard_error:
        return QueryResponse(
            answer=guard_error,
            provider="guardrail",
            model="none",
            sources=[],
            error=guard_error,
            cached=False,
            guardrail_triggered=True,
        )

    # Normalise force_provider — empty string means auto
    _force_provider = force_provider.strip() or None

    # Build a QueryRequest-like object for reuse in _process_query
    req = QueryRequest(
        query=clean_query or query or "",
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        force_provider=_force_provider,
        use_cache=False,  # Never cache file-upload queries
        use_hyde=use_hyde,
        use_query_rewrite=True,
        use_parent_child=use_parent_child,
        use_image_search=use_image_search,
    )

    # ── PDF processing path ───────────────────────────────────────────────────
    if is_pdf:
        try:
            import fitz  # type: ignore[import-not-found]
            import faiss as _faiss

            # Write to a temp file because fitz.open() needs a path
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)

            # Extract text from up to 10 pages to keep latency reasonable.
            page_texts = []
            for page_num in range(min(len(doc), 10)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    page_texts.append((page_num + 1, text))
            doc.close()

            try:
                os.unlink(tmp_path)
            except Exception:
                logger.warning(f"Failed to remove temp PDF file: {tmp_path}")

            if page_texts and embedding_model is not None and retriever.faiss_index is not None:
                for page_num, page_text in page_texts:
                    snippet = page_text[:2000]
                    try:
                        page_vec = embedding_model.embed_query(snippet).reshape(1, -1)
                        _faiss.normalize_L2(page_vec)
                        distances, indices = retriever.faiss_index.search(page_vec, top_k)
                        for dist, idx in zip(distances[0], indices[0]):
                            if 0 <= idx < len(retriever.faiss_chunks):
                                chunk = dict(retriever.faiss_chunks[idx])
                                chunk["_upload_match"] = True
                                chunk["_upload_source"] = f"<uploaded: {filename}, p.{page_num}>"
                                extra_text_chunks.append(chunk)
                    except Exception as page_err:
                        logger.warning(f"PDF page {page_num} embedding failed: {page_err}")

                logger.info(
                    f"📄 PDF upload '{filename}': {len(page_texts)} pages → "
                    f"{len(extra_text_chunks)} extra chunk matches"
                )
            else:
                logger.warning(
                    f"📄 PDF upload '{filename}': no text extracted or model/index unavailable"
                )
        except Exception as pdf_err:
            logger.error(f"❌ PDF processing failed for '{filename}': {pdf_err}")
            # Non-fatal — fall through to normal query processing

    # ── Image (PNG/JPEG/WEBP) processing path ─────────────────────────────────
    elif is_image:
        try:
            from PIL import Image as PILImage  # type: ignore[import-not-found]
            from pipelines.image_pipeline import get_clip_embedding

            pil_image = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
            clip_vector = get_clip_embedding(pil_image)   # returns list[float] | None

            if clip_vector and image_retriever is not None:
                import numpy as _np

                vec = _np.array(clip_vector, dtype="float32")
                img_results = image_retriever.retrieve_by_vector(vec, top_k=3)

                for img in img_results:
                    geo = img.get("geo_estimate", {})
                    city = geo.get("city") if geo else None
                    conf = geo.get("confidence", 0.0) if geo else 0.0
                    geo_str = f"{city} (conf {conf:.2f})" if city else "unknown"

                    extra_image_sources.append(
                        RetrievalResult(
                            chunk_id=img.get("image_path", ""),
                            text=img.get("caption", ""),
                            source_file=img.get("source_doc", filename),
                            page=0,
                            doc_type="IMAGE",
                            image_path=img.get("image_path"),
                            caption=img.get("caption"),
                            geo_city=city,
                            similarity=img.get("similarity"),
                        )
                    )

                logger.info(
                    f"🖼️  Image upload '{filename}': CLIP search → "
                    f"{len(extra_image_sources)} matching images"
                )
            else:
                logger.warning(
                    f"🖼️  Image upload '{filename}': "
                    f"CLIP embedding failed or image retriever not loaded"
                )

        except Exception as img_err:
            logger.error(f"❌ Image processing failed for '{filename}': {img_err}")
            # Non-fatal — fall through

    # ── Unsupported but syntactically-valid upload: log and continue ──────────
    else:
        logger.warning(f"⚠️  Unsupported file type: '{filename}' ({content_type})")

    # ── Run normal RAG pipeline ───────────────────────────────────────────────
    # If no text query was provided but a file was uploaded, synthesise a
    # generic query so the LLM has something to answer.
    if not req.query.strip():
        if is_pdf:
            req = req.model_copy(update={"query": f"Summarise the content of {filename}"})
        elif is_image:
            req = req.model_copy(
                update={
                    "query": f"Describe and find related content for the uploaded image {filename}"
                }
            )

    try:
        response = await asyncio.wait_for(_process_query(req), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Query-with-file error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # ── Merge upload-derived results into the response ────────────────────────

    # PDF extra chunks → prepend to sources list so they appear first
    if extra_text_chunks:
        upload_sources = [
            RetrievalResult(
                chunk_id=c.get("chunk_id", ""),
                text=c.get("text", "")[:200],
                source_file=c.get("_upload_source", c.get("source_file", filename)),
                page=c.get("page", 0),
                doc_type=c.get("doc_type", "uploaded_pdf"),
                confidence_score=0.0,
            )
            for c in extra_text_chunks[: req.rerank_top_n]
        ]
        existing_ids = {s.chunk_id for s in response.sources}
        new_sources = [s for s in upload_sources if s.chunk_id not in existing_ids]
        response = response.model_copy(update={"sources": new_sources + list(response.sources)})

    # Image upload extra matches → prepend to image_sources
    if extra_image_sources:
        existing_img_ids = {s.chunk_id for s in response.image_sources}
        new_img_sources = [
            s for s in extra_image_sources if s.chunk_id not in existing_img_ids
        ]
        response = response.model_copy(
            update={"image_sources": new_img_sources + list(response.image_sources)}
        )

    return response

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if retriever is None or router is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    # ── INPUT GUARDRAIL (unchanged – user's existing implementation) ──────
    classifier = _get_classifier_client()
    clean_query, guard_error = guard_input(req.query, classifier)

    if guard_error:
        logger.warning(f"Guardrail blocked: '{req.query[:80]}'")
        return QueryResponse(
            answer=guard_error,
            provider="guardrail",
            model="none",
            sources=[],
            error=guard_error,
            cached=False,
            guardrail_triggered=True,
        )

    req = req.model_copy(update={"query": clean_query})

    try:
        return await asyncio.wait_for(_process_query(req), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Query error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def _process_query(req: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline (guardrail already passed):
      1. Cache check
      2. Query rewriting
      3. HyDE document generation
      4a. Parent-Child retrieval (notebook Cell 18)  ← new optional path
      4b. Standard hybrid retrieval + cross-encoder reranking (original)
      5. LLM generation
      6. Cache store + return

    use_parent_child=True activates the notebook's AdvancedParentChildRetriever
    path; use_parent_child=False (default) uses the original fast path.
    """
    global _performance_metrics

    # ── 0. Record start time for performance tracking ───────────────────
    _query_start_time = datetime.now()

    # ── 1. Cache ─────────────────────────────────────────────────────────
    cache_key = "rag:" + hashlib.sha256(req.query.encode()).hexdigest()[:32]
    if req.use_cache and cache_key in cache:
        cached = cache[cache_key]
        cached["cached"] = True
        return QueryResponse(**cached)

    rewritten_query     = req.query
    hyde_used           = False
    hyde_embedding_text = req.query

    # ── 2. Query Rewriting ───────────────────────────────────────────────
    if req.use_query_rewrite:
        rewritten_query = router.rewrite_query(req.query)
        logger.info(f"   Rewritten: '{req.query}' → '{rewritten_query}'")

    # ── 3. HyDE ─────────────────────────────────────────────────────────
    if req.use_hyde:
        hyde_doc = router.generate_hyde_document(req.query)
        if hyde_doc:
            hyde_embedding_text = hyde_doc
            hyde_used           = True

    # ── 4a. Parent-Child retrieval (notebook Cell 18) ────────────────────
    if req.use_parent_child:
        pc_results = retriever.retrieve_with_parent_child(
            query           = rewritten_query,
            use_hyde        = req.use_hyde,
            use_multi_query = req.use_query_rewrite,
        )

        if pc_results:
            # Build context from parent content (richer than child alone)
            context_parts = [
                f"Source: {r.metadata.get('source_file', 'unknown')}\n{r.parent_content}"
                for r in pc_results[:req.rerank_top_n]
            ]
            context = "\n\n---\n\n".join(context_parts)

            # Map back to RetrievalResult pydantic model
            sources = [
                RetrievalResult(
                    chunk_id    = r.metadata.get("chunk_id", ""),
                    text        = r.child_content[:200],
                    source_file = r.metadata.get("source_file", ""),
                    page        = (r.metadata.get("page_numbers") or [0])[0],
                    doc_type    = r.metadata.get("doc_type", ""),
                    confidence_score = r.confidence_score,
                )
                for r in pc_results[: req.rerank_top_n]
            ]
            retrieved_chunks = [r.metadata for r in pc_results]
        else:
            context          = ""
            sources          = []
            retrieved_chunks = []

    # ── 4b. Standard hybrid retrieval + reranking (original path) ────────
    else:
        chunks = retriever.retrieve_and_rerank(
            query        = rewritten_query,
            faiss_query  = hyde_embedding_text,
            top_k        = req.top_k,
            rerank_top_n = req.rerank_top_n,
        )

        if chunks:
            sources = [
                RetrievalResult(
                    chunk_id    = c.get("chunk_id", ""),
                    text        = c.get("text", "")[:200],
                    source_file = c.get("source_file", ""),
                    page        = c.get("page", 0),
                    doc_type    = c.get("doc_type", ""),
                    confidence_score = 0.0,
                )
                for c in chunks
            ]
            retrieved_chunks = chunks
            context          = build_context(chunks)
        else:
            sources          = []
            retrieved_chunks = []
            context          = ""

    # ── 5. Image Retrieval (Gap Fix #2) ─────────────────────────────────
    # Run cross-modal CLIP search in parallel to text retrieval.
    # Uses the rewritten query so image search benefits from the same
    # query improvement as text search.
    # Image results are included in the LLM context so it can reference
    # visual evidence (captions, geo) alongside document text.
    image_sources: List[RetrievalResult] = []
    image_context_parts: List[str]       = []

    if req.use_image_search and image_retriever is not None:
        try:
            # Cap image results to max 8 regardless of text rerank_top_n to avoid too many images
            img_top_k = min(req.rerank_top_n, 8)
            img_results = image_retriever.retrieve(rewritten_query, top_k=img_top_k)

            for img in img_results:
                # Build a natural-language description of the image so the
                # LLM can reason about it without needing to see the pixels
                geo   = img.get("geo_estimate", {})
                city  = geo.get("city")    if geo else None
                conf  = geo.get("confidence", 0.0) if geo else 0.0
                geo_str = f"{city} (confidence {conf:.2f})" if city else "unknown location"

                img_desc = (
                    f"[IMAGE: {img.get('caption', 'unknown')} | "
                    f"Source: {img.get('source_doc', 'unknown')} | "
                    f"Geo: {geo_str} | "
                    f"Similarity: {img.get('similarity', 0.0):.3f}]"
                )
                image_context_parts.append(img_desc)

                image_sources.append(RetrievalResult(
                    chunk_id    = img.get("image_path", ""),
                    text        = img.get("caption", ""),
                    source_file = img.get("source_doc", ""),
                    page        = 0,
                    doc_type    = "IMAGE",
                    image_path  = img.get("image_path"),
                    caption     = img.get("caption"),
                    geo_city    = city,
                    similarity  = img.get("similarity"),
                ))

            if image_context_parts:
                # Append image context block to the text context so the LLM
                # sees both modalities in a single prompt
                image_block = "\n\nVISUAL EVIDENCE FROM IMAGES:\n" + "\n".join(image_context_parts)
                context += image_block
                logger.info(f"   🖼️  Added {len(image_context_parts)} image result(s) to context")

        except Exception as e:
            logger.warning(f"⚠️  Image retrieval failed: {e} — continuing with text only")

    # ── 6. LLM Generation ────────────────────────────────────────────────
    result = router.generate(
        context         = context,
        query           = req.query,
        retrieved_chunks= retrieved_chunks,
        force_provider  = req.force_provider,
    )

    logger.info(f"✅ {result['provider']} ({result['model']})")

    # ── 7. Track performance metrics (notebook Cell 28 style) ─────────────
    _performance_metrics.append({
        "timestamp":       datetime.now().isoformat(),
        "query":           req.query,
        "provider":        result["provider"],
        "retrieval_count": len(sources),
        "hyde_used":       hyde_used,
        "parent_child":    req.use_parent_child,
        "execution_time":  (datetime.now() - _query_start_time).total_seconds(),
    })
    # Cap in-memory metrics to the last 1000 entries to prevent unbounded growth
    if len(_performance_metrics) > 1000:
        del _performance_metrics[:-1000]

    # ── 8. Cache + Return ────────────────────────────────────────────────
    response = QueryResponse(
        answer          = result.get("answer", ""),
        provider        = result["provider"],
        model           = result["model"],
        sources         = sources,
        image_sources   = image_sources,          # Gap Fix #2: image results
        error           = result.get("error"),
        cached          = False,
        rewritten_query = rewritten_query if rewritten_query != req.query else None,
        hyde_used       = hyde_used,
        guardrail_triggered = False,
    )

    if req.use_cache:
        cache.set(cache_key, response.model_dump(), expire=3600)

    return response


@app.get("/retrieve")
async def retrieve_only(
    query:             str,
    top_k:             int  = 10,   # UI: Top K=10
    use_hyde:          bool = True, # UI: HyDE ON
    use_query_rewrite: bool = True,
    # Parent-Child retrieval OFF by default (UI toggle)
    use_parent_child:  bool = False,
    use_image_search:  bool = True,
):
    """Debug retrieval endpoint – returns text chunks, parent-child results, and image matches."""
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    # Guard this endpoint too (unchanged)
    classifier = _get_classifier_client()
    clean_query, guard_error = guard_input(query, classifier)
    if guard_error:
        return {"blocked": True, "message": guard_error}

    try:
        rewritten_query     = clean_query
        hyde_used           = False
        hyde_embedding_text = clean_query

        if use_query_rewrite:
            rewritten_query = router.rewrite_query(clean_query)
        if use_hyde:
            hyde_doc = router.generate_hyde_document(clean_query)
            if hyde_doc:
                hyde_embedding_text = hyde_doc
                hyde_used           = True

        if use_parent_child:
            # Notebook Cell 18 path
            pc_results = retriever.retrieve_with_parent_child(
                query           = rewritten_query,
                use_hyde        = use_hyde,
                use_multi_query = use_query_rewrite,
            )
            # Image retrieval — same query, runs regardless of text mode
            img_results = []
            if use_image_search and image_retriever is not None:
                try:
                    img_results = image_retriever.retrieve(rewritten_query)
                except Exception as _ie:
                    logger.warning(f"⚠️  /retrieve image search failed: {_ie}")

            return {
                "original_query":  query,
                "rewritten_query": rewritten_query,
                "hyde_used":       hyde_used,
                "parent_child":    True,
                "num_results":     len(pc_results),
                "num_image_results": len(img_results),
                "results": [
                    {
                        "chunk_id":       r.metadata.get("chunk_id", ""),
                        "child_text":     r.child_content[:300],
                        "parent_text":    r.parent_content[:300],
                        "confidence":     r.confidence_score,
                        "method":         r.retrieval_method,
                        "source":         r.metadata.get("source_file", ""),
                        "page":           (r.metadata.get("page_numbers") or [0])[0],
                        "source_type":    "TEXT",
                    }
                    for r in pc_results
                ],
                "image_results": [
                    {
                        "image_path": img.get("image_path", ""),
                        "source_doc": img.get("source_doc", ""),
                        "caption":    img.get("caption", ""),
                        "geo":        img.get("geo_estimate", {}),
                        "similarity": round(img.get("similarity", 0.0), 4),
                    }
                    for img in img_results
                ],
            }
        else:
            # Original path
            chunks = retriever.retrieve_and_rerank(
                query        = rewritten_query,
                faiss_query  = hyde_embedding_text,
                top_k        = top_k,
                rerank_top_n = top_k,
            )
            # Image retrieval — same query, runs regardless of text mode
            img_results = []
            if use_image_search and image_retriever is not None:
                try:
                    img_results = image_retriever.retrieve(rewritten_query)
                except Exception as _ie:
                    logger.warning(f"⚠️  /retrieve image search failed: {_ie}")

            return {
                "original_query":  query,
                "rewritten_query": rewritten_query,
                "hyde_used":       hyde_used,
                "parent_child":    False,
                "num_results":     len(chunks),
                "num_image_results": len(img_results),
                "results": [
                    {
                        "chunk_id":   c.get("chunk_id", ""),
                        "text":       c.get("text", "")[:300],
                        "source":     c.get("source_file", ""),
                        "page":       c.get("page", 0),
                        "source_type": "TEXT",
                    }
                    for c in chunks
                ],
                "image_results": [
                    {
                        "image_path": img.get("image_path", ""),
                        "source_doc": img.get("source_doc", ""),
                        "caption":    img.get("caption", ""),
                        "geo":        img.get("geo_estimate", {}),
                        "similarity": round(img.get("similarity", 0.0), 4),
                    }
                    for img in img_results
                ],
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def system_status():
    """
    System health and performance metrics (notebook Cell 28 get_system_status pattern).
    Returns component health, FAISS stats, and query performance averages.
    """
    try:
        faiss_total = retriever.faiss_index.ntotal if retriever and retriever.faiss_index else 0
        bm25_total  = len(retriever.bm25_chunks)   if retriever else 0
        parent_count = len(retriever.parent_store)  if retriever else 0

        avg_time = (
            float(np.mean([m.get("execution_time", 0) for m in _performance_metrics]))
            if _performance_metrics else 0.0
        )

        img_vectors = (
            image_retriever.image_faiss_index.ntotal
            if image_retriever and image_retriever.image_faiss_index else 0
        )
        img_records = len(image_retriever.image_chunks) if image_retriever else 0

        return {
            "system_health":   "healthy" if retriever and router else "degraded",
            "timestamp":       datetime.now().isoformat(),
            "vector_store": {
                "faiss_vectors":    faiss_total,
                "bm25_chunks":      bm25_total,
                "parent_chunks":    parent_count,
                "embedding_model":  EMBEDDING_MODEL,
                "image_vectors":    img_vectors,
                "image_records":    img_records,
            },
            "performance_metrics": {
                "total_queries":        len(_performance_metrics),
                "avg_execution_time":   round(avg_time, 3),
            },
            "components": {
                "embedding_model":   "active" if embedding_model else "offline",
                "faiss_index":       "active" if (retriever and retriever.faiss_index) else "offline",
                "bm25_index":        "active" if (retriever and retriever.bm25_index) else "offline",
                "image_index":       "active" if img_vectors > 0 else "offline",
                "llm_router":        "active" if router else "offline",
                "security_layer":    "active",
            },
        }
    except Exception as e:
        return {"system_health": "degraded", "error": str(e)}


@app.post("/cache/clear")
async def clear_cache():
    cache.clear()
    return {"status": "Cache cleared", "timestamp": datetime.now().isoformat()}


@app.get("/cache/stats")
async def cache_stats():
    return {"cache_dir": CACHE_DIR, "size": len(cache)}


@app.get("/")
async def root():
    return {
        "name":        "MultiModal RAG API",
        "version":     "1.0.0",
        "guardrails":  "Input-only, balanced — Layer1 hard rules + Layer2 LLM classifier",
        "new_features": [
            "Parent-Child retrieval (use_parent_child=True)",
            "Confidence scoring per retrieved chunk",
            "/status endpoint with system health metrics",
        ],
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )