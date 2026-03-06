"""
Embeddings Module: Build FAISS and BM25 indexes for hybrid retrieval
=====================================================================
Upgraded to incorporate the Advanced RAG notebook's patterns:
  - EmbeddingManager with embed_texts / embed_query helpers (notebook Cell 14 style)
  - Batch processing with progress bar and stats logging (notebook Cell 14)
  - IndexLoader, FAISSIndexBuilder, BM25IndexBuilder kept and enhanced
  - Device detection (CUDA / CPU) kept from original
  - BM25 k1/b tuning comments kept from original
"""

import json
import os
import pickle
import tempfile
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

from lib.file_resolver import IS_PRODUCTION, resolve_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Device detection ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding device: {DEVICE}")

# ── BM25 Tuning Parameters ──────────────────────────────────────────────────
# k1: Term frequency saturation factor [1.2–2.0]
#   Lower (0.5–0.9): diminishing returns after ~2 occurrences of a term
#   Higher (1.5–2.0): keeps rewarding repeated terms more
BM25_K1 = 1.5

# b: Document length normalisation [0.0–1.0]
#   0.0 = no normalisation (large docs not penalised)
#   1.0 = full normalisation (strongly favours short docs)
#   0.75 = balanced: slight preference for shorter docs
BM25_B = 0.75


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingManager:
    """
    Manage text embeddings (notebook Cell 14 pattern + original project needs).

    Changes vs original:
      • __init__ logs model load just like notebook EnterpriseVectorStore does
      • embed_texts uses 'passage: ' prefix required by e5 models (unchanged)
      • embed_query uses 'query: ' prefix (unchanged)
      • Added embed_batch helper that matches notebook's batch-with-stats pattern
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        """
        Initialise embedding model.

        Model options (CPU-friendly, best→fastest):
          intfloat/e5-base-v2    – 768-dim, ~2× faster than e5-large  ← default
          intfloat/e5-small-v2   – 384-dim, ~4× faster
          BAAI/bge-small-en-v1.5 – 384-dim, very fast, strong retrieval
        """
        self.model_name = model_name
        logger.info(f"🚀 Loading embedding model: {model_name} on {DEVICE}")
        self.model         = SentenceTransformer(model_name, device=DEVICE)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"   Embedding dimension: {self.embedding_dim}")

    # ── Core encode methods ────────────────────────────────────────────────

    def embed_texts(
        self,
        texts:      List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Embed a list of passage texts (notebook Cell 14 pattern).
        Prepends 'passage: ' required by e5 models.
        Larger batch_size (64) speeds up both CPU and GPU vs default 32.
        """
        prefixed = [f"passage: {t}" for t in texts]

        # notebook-style info logging before encode
        logger.info(f"🔄 Embedding {len(texts)} texts | batch_size={batch_size}")

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        logger.info(f"✅ Embeddings ready: shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query with 'query: ' prefix (unchanged from original).
        Used by HybridRetriever at inference time.
        """
        return self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_batch(
        self,
        texts:      List[str],
        batch_size: int = 100,
        prefix:     str = "passage: ",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Batch embedding with per-batch stats (inspired by notebook Cell 14
        add_chunks batch loop that tracks successful/failed counts).

        Returns:
            embeddings – np.ndarray (n, dim)
            stats      – dict with total, batches_processed, failed
        """
        stats = {
            "total_texts":       len(texts),
            "batches_processed": 0,
            "failed_batches":    0,
        }

        all_embeddings = []
        total_batches  = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch     = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                prefixed = [f"{prefix}{t}" for t in batch]
                embs     = self.model.encode(
                    prefixed,
                    batch_size=min(batch_size, 32),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype("float32")

                all_embeddings.append(embs)
                stats["batches_processed"] += 1
                logger.debug(f"   Batch {batch_num}/{total_batches}: {len(batch)} texts ✅")

            except Exception as e:
                logger.error(f"   ❌ Batch {batch_num} failed: {e}")
                stats["failed_batches"] += 1

        if not all_embeddings:
            raise RuntimeError("All embedding batches failed")

        combined = np.vstack(all_embeddings)
        logger.info(f"✅ Batch embedding done | shape={combined.shape} | "
                    f"batches={stats['batches_processed']}/{total_batches}")
        return combined, stats


# ═══════════════════════════════════════════════════════════════════════════
# FAISS INDEX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class FAISSIndexBuilder:
    """
    Build and persist FAISS indexes (unchanged structure, added logging style).

    Changes vs original:
      • Log messages use notebook emoji style for consistency
      • IVF nlist calculation and nprobe unchanged
    """

    @staticmethod
    def build_index(
        embeddings:  np.ndarray,
        chunks:      List[Dict],
        output_dir:  str,
        index_type:  str = "ivf",
        nlist:       Optional[int] = None,
    ):
        """
        Build IVF FAISS index for efficient retrieval.

        Args:
            embeddings:  (n, dim) float32 array
            chunks:      list of chunk dicts (saved alongside index)
            output_dir:  directory to write index.bin and chunks.json
            index_type:  'ivf' (recommended) or 'flat'
            nlist:       IVF cluster count; auto-calculated if None
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n, d = embeddings.shape
        logger.info(f"🔧 Building FAISS index | vectors={n}, dim={d}, type={index_type}")

        if index_type == "ivf":
            # Auto-select nlist: keep between 4 and 256 (original formula)
            if nlist is None:
                nlist = min(256, max(n // 10, 4))

            quantizer = faiss.IndexFlatIP(d)
            index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(32, nlist // 4)

            logger.info(f"   IVF: nlist={nlist}, nprobe={index.nprobe}")

        else:
            # Flat index – exact but slow for large corpora
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            logger.info("   Flat index (exact search)")

        # Persist index
        index_path = output_dir / "index.bin"
        faiss.write_index(index, str(index_path))
        logger.info(f"✅ FAISS index saved → {index_path}")

        # Persist chunk metadata alongside
        chunks_path = output_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Chunk metadata saved → {chunks_path}")

        return index


# ═══════════════════════════════════════════════════════════════════════════
# BM25 INDEX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class BM25IndexBuilder:
    """
    Build and persist BM25 keyword indexes (unchanged from original, logging upgraded).
    """

    @staticmethod
    def build_index(
        chunks:      List[Dict],
        output_path: str,
        k1:          float = BM25_K1,
        b:           float = BM25_B,
    ):
        """
        Build BM25Okapi index.

        BM25 formula per term t in query Q for document d:
          score(d,Q) = Σ IDF(t) × (TF(t,d)×(k1+1)) / (TF(t,d) + k1×(1−b + b×|d|/avgdl))

        Args:
            chunks:       list of chunk dicts with 'text' field
            output_path:  path to write pickled index
            k1:           term-frequency saturation [1.2–2.0]
            b:            length normalisation [0.0–1.0]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔧 Building BM25 index | chunks={len(chunks)}, k1={k1}, b={b}")
        logger.info(
            f"   k1={k1}: {'low saturation' if k1 < 1.2 else 'standard saturation'}"
        )
        logger.info(
            f"   b={b}: {'no length norm' if b == 0 else 'full length norm' if b == 1 else 'balanced length norm'}"
        )

        tokenized = [c.get("text", "").lower().split() for c in chunks]
        bm25      = BM25Okapi(tokenized, k1=k1, b=b)

        with open(output_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks, "params": {"k1": k1, "b": b}}, f)

        logger.info(f"✅ BM25 index saved → {output_path}")
        return bm25


# ═══════════════════════════════════════════════════════════════════════════
# INDEX LOADER
# ═══════════════════════════════════════════════════════════════════════════

class IndexLoader:
    """Load persisted FAISS and BM25 indexes (local in dev, R2 in production)."""

    @staticmethod
    def to_r2_key(path: Path | str) -> str:
        """
        Convert a local filesystem path to an R2 object key.

        - Uses forward slashes on all platforms
        - Strips leading './' so keys are stable
        """
        return str(path).replace("\\", "/").lstrip("./")

    @staticmethod
    def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
        """Load FAISS index and associated chunk metadata."""
        index_dir_path = Path(index_dir)
        logger.info(f"📂 Loading FAISS index from {index_dir_path}")

        index_path_local = index_dir_path / "index.bin"
        chunks_path_local = index_dir_path / "chunks.json"

        # Development: prefer local filesystem for speed
        if not IS_PRODUCTION and index_path_local.exists() and chunks_path_local.exists():
            index = faiss.read_index(str(index_path_local))
            with open(chunks_path_local, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info(f"✅ FAISS loaded (local) | vectors={index.ntotal}, chunks={len(chunks)}")
            return index, chunks

        # Production or missing local files: load from Cloudflare R2 via file_resolver
        key_prefix = IndexLoader.to_r2_key(index_dir_path)
        index_key = f"{key_prefix}/index.bin"
        chunks_key = f"{key_prefix}/chunks.json"

        logger.info(f"📂 Loading FAISS index from R2 key prefix '{key_prefix}'")

        index_bytes = resolve_file(index_key)
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            tmp.write(index_bytes)
            tmp_path = tmp.name
        try:
            index = faiss.read_index(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning(f"Failed to remove temp FAISS file: {tmp_path}")

        chunks_bytes = resolve_file(chunks_key)
        chunks = json.loads(chunks_bytes.decode("utf-8"))

        logger.info(f"✅ FAISS loaded (R2) | vectors={index.ntotal}, chunks={len(chunks)}")
        return index, chunks

    @staticmethod
    def load_bm25_index(index_path: str) -> Tuple[BM25Okapi, List[Dict]]:
        """Load BM25 index and associated chunk list."""
        index_path_obj = Path(index_path)
        logger.info(f"📂 Loading BM25 index from {index_path_obj}")

        # Development: prefer local file when available
        if not IS_PRODUCTION and index_path_obj.exists():
            with open(index_path_obj, "rb") as f:
                data = pickle.load(f)
        else:
            key = IndexLoader.to_r2_key(index_path_obj)
            logger.info(f"📂 Loading BM25 index from R2 key '{key}'")
            data_bytes = resolve_file(key)
            data = pickle.loads(data_bytes)

        params = data.get("params", {})
        logger.info(
            f"✅ BM25 loaded | chunks={len(data['chunks'])}, "
            f"k1={params.get('k1', '?')}, b={params.get('b', '?')}"
        )
        return data["bm25"], data["chunks"]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BUILD FUNCTION  (used by orchestrate.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_text_indexes(
    chunks_json_path:     str,
    output_dir:           str,
    embedding_model_name: str   = "intfloat/e5-small-v2",
    bm25_k1:              float = BM25_K1,
    bm25_b:               float = BM25_B,
) -> dict:
    """
    Load combined chunks JSON → embed → build FAISS + BM25 indexes.

    Called by orchestrate.py --embed phase.
    Signature unchanged; adds notebook-style stats logging.

    Args:
        chunks_json_path:     path to deduplicated _all_chunks.json
        output_dir:           where to write vectorstore files
        embedding_model_name: sentence-transformers model
        bm25_k1:              BM25 term-frequency saturation
        bm25_b:               BM25 length normalisation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Loading chunks from {chunks_json_path}")
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"   Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    # ── Embed (notebook Cell 14 style with stats) ────────────────────────
    embedding_mgr = EmbeddingManager(embedding_model_name)
    embeddings, embed_stats = embedding_mgr.embed_batch(texts, batch_size=64)

    logger.info(f"   Embedding stats: {embed_stats}")

    # ── FAISS index ──────────────────────────────────────────────────────
    faiss_dir = output_dir / "faiss"
    FAISSIndexBuilder.build_index(embeddings, chunks, str(faiss_dir))

    # ── BM25 index (tuned k1/b) ──────────────────────────────────────────
    bm25_path = output_dir / "bm25.pkl"
    BM25IndexBuilder.build_index(chunks, str(bm25_path), k1=bm25_k1, b=bm25_b)

    logger.info("🎉 All indexes built successfully.")

    return {
        "faiss_dir":  str(faiss_dir),
        "bm25_path":  str(bm25_path),
        "num_chunks": len(chunks),
        "model":      embedding_model_name,
        "device":     DEVICE,
        "bm25_k1":    bm25_k1,
        "bm25_b":     bm25_b,
        # New: surface embedding stats for pipeline_results.json
        "embed_stats": embed_stats,
    }



# ═══════════════════════════════════════════════════════════════════════════
# IMAGE INDEX BUILDER  (Gap Fix #1)
# ═══════════════════════════════════════════════════════════════════════════

def build_image_indexes(
    image_metadata_path: str,
    output_dir:          str,
) -> dict:
    """
    Build a CLIP-space FAISS index from image_metadata.json produced by
    image_pipeline.py.

    Why a separate index
    --------------------
    Image embeddings from CLIP (512-dim, ViT-B/32) live in a completely
    different vector space from text embeddings produced by e5-base-v2
    (768-dim).  Mixing them in a single FAISS index would be meaningless —
    dimensions don't match and the spaces are not aligned.  We therefore
    build a dedicated 512-dim image FAISS index that ImageRetriever
    (retriever.py) queries at inference time using CLIP text embeddings.

    Args:
        image_metadata_path:  path to processed/image_meta/image_metadata.json
                              (the full version that includes "embedding" arrays)
        output_dir:           where to write image_faiss/index.bin +
                              image_faiss/chunks.json

    Returns:
        dict with index path, chunk count, and dimension info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Loading image metadata from {image_metadata_path}")
    with open(image_metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"   Loaded {len(records)} image records")

    # Filter records that have valid embeddings
    valid = [r for r in records if r.get("embedding") and len(r["embedding"]) > 0]
    if not valid:
        raise ValueError("No valid image embeddings found in metadata file")

    skipped = len(records) - len(valid)
    if skipped:
        logger.warning(f"   ⚠️  Skipped {skipped} records with missing embeddings")

    # Stack embeddings into (n, dim) float32 array
    embeddings = np.array([r["embedding"] for r in valid], dtype="float32")
    n, d       = embeddings.shape
    logger.info(f"   Embedding matrix: shape=({n}, {d})")

    # Verify all embeddings are L2-normalised (they should be from image_pipeline.py)
    # Re-normalise defensively to avoid cosine-space errors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    # Build FAISS index — use Flat for image collections (typically small)
    # IVF requires nlist <= n/10; most image sets have <10k images so Flat
    # gives exact search with negligible overhead.
    if n >= 100:
        nlist     = min(256, max(n // 10, 4))
        quantizer = faiss.IndexFlatIP(d)
        index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(16, nlist // 4)
        index_type   = f"IVF(nlist={nlist}, nprobe={index.nprobe})"
    else:
        index      = faiss.IndexFlatIP(d)
        index.add(embeddings)
        index_type = "Flat"

    logger.info(f"   Index type: {index_type}")

    # Persist index
    image_faiss_dir = output_dir / "image_faiss"
    image_faiss_dir.mkdir(parents=True, exist_ok=True)

    index_path = image_faiss_dir / "index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"✅ Image FAISS index saved → {index_path}")

    # Persist slim chunk metadata (drop embedding array to save space —
    # FAISS stores the vectors; we only need the metadata for lookup)
    slim_records = [
        {k: v for k, v in r.items() if k != "embedding"}
        for r in valid
    ]
    chunks_path = image_faiss_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(slim_records, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Image chunk metadata saved → {chunks_path}")

    return {
        "image_faiss_dir": str(image_faiss_dir),
        "num_images":      n,
        "embedding_dim":   d,
        "index_type":      index_type,
        "model":           "openai/clip-vit-base-patch32",
    }


class ImageIndexLoader:
    """Load a persisted image FAISS index and its chunk metadata."""

    @staticmethod
    def load_image_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
        """
        Load image FAISS index built by build_image_indexes().

        Args:
            index_dir: directory containing index.bin and chunks.json

        Returns:
            (faiss.Index, list-of-image-chunk-dicts)
        """
        index_dir_path = Path(index_dir)
        logger.info(f"📂 Loading image FAISS index from {index_dir_path}")

        index_path_local = index_dir_path / "index.bin"
        chunks_path_local = index_dir_path / "chunks.json"

        # Development: read from local disk when present
        if not IS_PRODUCTION and index_path_local.exists() and chunks_path_local.exists():
            index = faiss.read_index(str(index_path_local))
            with open(chunks_path_local, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info(
                f"✅ Image FAISS loaded (local) | vectors={index.ntotal}, "
                f"dim={index.d}, records={len(chunks)}"
            )
            return index, chunks

        # Production or missing local files: fetch from R2
        key_prefix = IndexLoader.to_r2_key(index_dir_path)
        index_key = f"{key_prefix}/index.bin"
        chunks_key = f"{key_prefix}/chunks.json"

        logger.info(f"📂 Loading image FAISS index from R2 key prefix '{key_prefix}'")

        index_bytes = resolve_file(index_key)
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            tmp.write(index_bytes)
            tmp_path = tmp.name
        try:
            index = faiss.read_index(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning(f"Failed to remove temp image FAISS file: {tmp_path}")

        chunks_bytes = resolve_file(chunks_key)
        chunks = json.loads(chunks_bytes.decode("utf-8"))

        logger.info(
            f"✅ Image FAISS loaded (R2) | vectors={index.ntotal}, "
            f"dim={index.d}, records={len(chunks)}"
        )
        return index, chunks


# ── CLI convenience ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = build_text_indexes(
            chunks_json_path=sys.argv[1],
            output_dir=sys.argv[2] if len(sys.argv) > 2 else "./vectorstore",
        )
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python embeddings.py <chunks_json> [output_dir]")