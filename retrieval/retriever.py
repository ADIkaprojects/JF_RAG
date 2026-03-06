"""
Retriever Module: Hybrid BM25 + FAISS with Advanced RAG techniques
===================================================================
Upgraded to incorporate the Advanced RAG notebook's patterns (Cell 18):
  - ParentChunk / RetrievalResult dataclasses (notebook Cell 18)
  - HyDE (Hypothetical Document Embeddings) generation (notebook Cell 18)
  - Multi-query expansion with LLM (notebook Cell 18)
  - _calculate_confidence per retrieval method (notebook Cell 18)
  - _rank_and_deduplicate_results by parent (notebook Cell 18)
  - retrieve_with_parent_child full pipeline (notebook Cell 18)

Original project features kept:
  - HybridRetriever with BM25 + FAISS + RRF fusion
  - Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - retrieve_and_rerank entry-point used by main.py / router.py
  - build_context helper used by main.py
"""

import os
import numpy as np
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES  (notebook Cell 18)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParentChunk:
    """
    Larger context chunk that contains multiple child chunks (notebook Cell 18).
    Stored in-memory in HybridRetriever.parent_store for context expansion.
    """
    parent_id:       str
    content:         str
    document_id:     str
    child_chunk_ids: List[str]  = field(default_factory=list)
    start_char:      int        = 0
    end_char:        int        = 0
    page_numbers:    List[int]  = field(default_factory=list)
    summary:         Optional[str] = None


@dataclass
class RetrievalResult:
    """
    Enhanced retrieval result carrying both child and parent context (notebook Cell 18).
    Used by retrieve_with_parent_child; retrieve_and_rerank returns plain dicts
    for backwards compatibility with main.py.
    """
    child_content:   str
    parent_content:  str
    similarity_score: float
    metadata:        Dict[str, Any]
    # 'original' | 'hyde' | 'multi_query' | 'bm25' | 'faiss'
    retrieval_method: str
    confidence_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Full-featured retriever combining BM25 + FAISS with advanced techniques.

    Architecture (notebook Cell 18 + original project):
    ┌─────────────────────────────────────────────────────┐
    │  retrieve_and_rerank  (main.py interface – unchanged)│
    │    └─ retrieve_hybrid  (BM25 + FAISS + RRF)          │
    │         ├─ retrieve_bm25   (keyword search)          │
    │         └─ retrieve_faiss  (dense semantic search)   │
    │    └─ rerank  (cross-encoder)                        │
    ├─────────────────────────────────────────────────────┤
    │  retrieve_with_parent_child  (notebook pipeline)     │
    │    ├─ generate_hyde_query    (HyDE, notebook Cell 18)│
    │    ├─ generate_multi_queries (notebook Cell 18)      │
    │    └─ _rank_and_deduplicate_results (notebook Cell 18)│
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        faiss_index,
        faiss_chunks,
        bm25_index,
        bm25_chunks,
        embedding_model,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_client=None,
        parent_chunk_size: int = 1200,
        child_chunk_size: int = 400,
        retrieval_count: int = 50,
    ):
        """
        Initialise HybridRetriever with text FAISS + BM25 indexes and reranker.

        This signature is relied on by api.main.startup_event, which passes in
        pre-loaded FAISS / BM25 indexes, the EmbeddingManager, and an LLMRouter
        instance for HyDE + multi-query generation.
        """
        # Core indexes / stores
        self.faiss_index   = faiss_index
        self.faiss_chunks  = faiss_chunks or []
        self.bm25_index    = bm25_index
        self.bm25_chunks   = bm25_chunks or []
        self.embedding_model = embedding_model

        # LLM router used for HyDE + multi-query (Advanced RAG notebook Cell 18)
        self.llm_client      = llm_client

        # Parent-child configuration
        self.parent_store:    Dict[str, ParentChunk] = {}
        self.child_to_parent: Dict[str, str]        = {}
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size  = child_chunk_size
        self.retrieval_count   = retrieval_count

        logger.info(f"🚀 Loading reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        logger.info("✅ HybridRetriever initialised")
        logger.info(f"   FAISS chunks: {len(self.faiss_chunks)}")
        logger.info(f"   BM25  chunks: {len(self.bm25_chunks)}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 – ORIGINAL PROJECT: BM25 + FAISS + RRF + Cross-encoder
    # ═══════════════════════════════════════════════════════════════════════

    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        BM25 keyword search (unchanged from original).
        Uses the rewritten query (clean terms, no filler words).

        BM25 score per document d for query Q:
          score(d,Q) = Σ IDF(t) × [TF(t,d)×(k1+1)] / [TF(t,d) + k1×(1−b + b×|d|/avgdl)]
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not loaded")
            return []

        tokens      = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        return [
            (self.bm25_chunks[i], float(bm25_scores[i]))
            for i in top_indices
            if bm25_scores[i] > 0
        ]

    def retrieve_faiss(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Dict, float]]:
        """
        FAISS dense semantic search (unchanged from original).
        Accepts either the rewritten query or a HyDE document as input –
        the caller decides which to pass via faiss_query in retrieve_hybrid.
        """
        if self.faiss_index is None or self.embedding_model is None:
            logger.warning("FAISS index or embedding model not loaded")
            return []

        # embed_query uses 'query: ' prefix (correct for e5 models at retrieval time)
        # embed_texts uses 'passage: ' prefix and is only correct during indexing
        query_vec = self.embedding_model.embed_query(query).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vec, top_k)

        results = []
        for j, i in enumerate(indices[0]):
            if i < len(self.faiss_chunks):
                chunk = dict(self.faiss_chunks[i])   # shallow copy to avoid mutating stored data
                chunk["_faiss_score"] = float(distances[0][j])
                results.append((chunk, float(distances[0][j])))
        return results

    def reciprocal_rank_fusion(
        self,
        bm25_results:  List[Tuple[Dict, float]],
        faiss_results: List[Tuple[Dict, float]],
        k: int = 60,
    ) -> List[Dict]:
        """
        RRF fusion (unchanged from original).

        RRF_score(d) = Σ 1 / (k + rank(d))

        k=60 is the standard constant that dampens very-high-rank influence.
        Scale-invariant: works regardless of BM25 vs FAISS score magnitudes.
        """
        rrf_scores: Dict[str, float] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk.get("chunk_id", id(chunk))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1 / (k + rank + 1)

        for rank, (chunk, _) in enumerate(faiss_results):
            cid = chunk.get("chunk_id", id(chunk))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1 / (k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        # Build lookup from the scored copies (faiss results carry _faiss_score;
        # faiss entries overwrite bm25 entries so that key is preserved).
        all_chunks = {
            chunk.get("chunk_id", id(chunk)): chunk
            for chunk, _ in bm25_results + faiss_results
        }

        return [all_chunks[cid] for cid in sorted_ids if cid in all_chunks]

    def retrieve_hybrid(
        self,
        query:       str,
        faiss_query: Optional[str] = None,
        top_k:       int = 10,
    ) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + FAISS with RRF fusion (unchanged from original).

        Args:
            query:        rewritten query for BM25 keyword search
            faiss_query:  HyDE doc (or fallback to query) for FAISS dense search
            top_k:        candidates from each index before fusion
        """
        faiss_query = faiss_query or query

        bm25_results  = self.retrieve_bm25(query, top_k=top_k)
        faiss_results = self.retrieve_faiss(faiss_query, top_k=top_k)

        logger.info(
            f"BM25: {len(bm25_results)} | "
            f"FAISS: {len(faiss_results)} | "
            f"HyDE active: {faiss_query != query}"
        )

        fused = self.reciprocal_rank_fusion(bm25_results, faiss_results, k=60)
        return fused[:top_k]

    def rerank(
        self,
        query:      str,
        candidates: List[Dict],
        top_n:      int = 10,
    ) -> List[Dict]:
        """
        Cross-encoder reranking (unchanged from original).
        Always uses the ORIGINAL user query (not HyDE/rewritten) so the
        reranker scores against what the user actually asked.
        """
        if not candidates:
            return []

        logger.info(f"🔍 Reranking {len(candidates)} candidates…")

        pairs  = [(query, c.get("text", "")) for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]

    def retrieve_and_rerank(
        self,
        query:        str,
        faiss_query:  Optional[str] = None,
        top_k:        int = 20,
        rerank_top_n: int = 10,
    ) -> List[Dict]:
        """
        Full pipeline for main.py (unchanged interface):
          1. Hybrid retrieval (BM25 + FAISS + RRF)
          2. Cross-encoder reranking

        Args:
            query:        rewritten query for BM25 / reranker
            faiss_query:  HyDE doc for FAISS dense search
            top_k:        candidates per index
            rerank_top_n: final results after reranking
        """
        candidates = self.retrieve_hybrid(
            query=query,
            faiss_query=faiss_query,
            top_k=top_k,
        )

        if not candidates:
            logger.warning("No candidates retrieved")
            return []

        # Rerank all candidates, then slice to rerank_top_n
        results = self.rerank(query, candidates, top_n=max(rerank_top_n, 20))
        logger.info(f"✅ Final results after reranking: {len(results)}")
        return results[:rerank_top_n]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 – NOTEBOOK Cell 18: Parent-Child + HyDE + Multi-Query
    # ═══════════════════════════════════════════════════════════════════════

    def generate_hyde_query(self, original_query: str) -> str:
        """
        Generate a hypothetical answer document (HyDE technique, notebook Cell 18).

        The HyDE doc is embedded instead of the bare query for FAISS search –
        a hypothetical answer lives in the same vector space as stored passages,
        improving recall on sparse or indirect queries.

        Falls back to the original query if the LLM is unavailable.
        """
        logger.debug(f"🔮 Generating HyDE for: '{original_query}'")

        # Use the LLMRouter's hyde generator if it was wired in
        if self.llm_client is not None:
            try:
                hyde_doc = self.llm_client.generate_hyde_document(original_query)
                if hyde_doc:
                    logger.debug(f"   HyDE doc: '{hyde_doc[:80]}…'")
                    return hyde_doc
            except Exception as e:
                logger.warning(f"⚠️  HyDE generation failed: {e}. Using original query.")

        return original_query

    def generate_multi_queries(
        self,
        original_query: str,
        num_queries:    int = 3,
    ) -> List[str]:
        """
        Generate multiple query variations for comprehensive retrieval (notebook Cell 18).

        Each variation uses different vocabulary while maintaining the same meaning.
        Falls back to a single-item list if the LLM is unavailable.
        """
        logger.debug(f"🔄 Generating {num_queries} query variations for: '{original_query}'")

        if self.llm_client is not None:
            try:
                rewritten = self.llm_client.rewrite_query(original_query)
                # Basic variation: rewritten + original forms two of the N
                variations = [rewritten, original_query]
                # If the router exposes a multi-query method use it; otherwise stop here
                if hasattr(self.llm_client, "generate_multi_queries"):
                    variations = self.llm_client.generate_multi_queries(
                        original_query, num_queries=num_queries
                    )
                return [v for v in variations if len(v) > 10][:num_queries]
            except Exception as e:
                logger.warning(f"⚠️  Multi-query generation failed: {e}. Using original only.")

        return [original_query]

    def retrieve_with_parent_child(
        self,
        query:           str,
        use_hyde:        bool = True,
        use_multi_query: bool = True,
    ) -> List[RetrievalResult]:
        """
        Advanced retrieval pipeline (notebook Cell 18).

        Steps:
          1. Optionally generate HyDE doc for dense retrieval
          2. Optionally expand query into multiple variations
          3. Run hybrid BM25 + FAISS for each query variant
          4. Attach parent context for each retrieved child chunk
          5. Rank and deduplicate by parent, return top results

        Args:
            query:           user's original query (used for reranking)
            use_hyde:        whether to generate HyDE doc for FAISS
            use_multi_query: whether to expand into multiple queries
        """
        logger.info(f"🔍 Parent-Child retrieval | query='{query}'")

        # Build query list to run
        queries_to_process: List[Tuple[str, str]] = []

        if use_hyde:
            hyde_q = self.generate_hyde_query(query)
            queries_to_process.append((hyde_q, "hyde"))

        if use_multi_query:
            for mq in self.generate_multi_queries(query):
                queries_to_process.append((mq, "multi_query"))

        # Always include the original query
        queries_to_process.append((query, "original"))

        logger.debug(f"   Processing {len(queries_to_process)} query variants")

        all_results: List[RetrievalResult] = []

        for query_text, method in queries_to_process:
            try:
                # For FAISS we use the query_text directly (it's already HyDE or rewritten)
                faiss_q      = query_text
                bm25_q       = query_text if method != "hyde" else query
                candidates   = self.retrieve_hybrid(
                    query=bm25_q,
                    faiss_query=faiss_q,
                    top_k=self.retrieval_count,
                )

                for chunk in candidates:
                    child_id = chunk.get("chunk_id", "")

                    # Look up parent context if the parent-child index is populated
                    if child_id in self.child_to_parent:
                        parent_id    = self.child_to_parent[child_id]
                        parent_chunk = self.parent_store.get(parent_id)
                        parent_content = parent_chunk.content if parent_chunk else chunk.get("text", "")
                        parent_pages   = parent_chunk.page_numbers if parent_chunk else []
                    else:
                        # No parent-child index: treat the chunk itself as its own context
                        parent_id      = child_id
                        parent_content = chunk.get("text", "")
                        parent_pages   = [chunk.get("page", 0)]

                    # Similarity score — use the FAISS inner-product distance stored
                    # by retrieve_faiss() in the '_faiss_score' key on each result.
                    # retrieve_hybrid() fuses BM25+FAISS results but strips this key,
                    # so we fall back to reading from the FAISS result directly if present,
                    # otherwise use the RRF-fused score stored as '_rrf_score'.
                    sim_score = chunk.get("_faiss_score") or chunk.get("_rrf_score") or 0.5

                    rr = RetrievalResult(
                        child_content    = chunk.get("text", ""),
                        parent_content   = parent_content,
                        similarity_score = sim_score,
                        metadata         = {
                            **chunk,
                            "parent_id":    parent_id,
                            "query_text":   query_text,
                            "page_numbers": parent_pages,
                        },
                        retrieval_method = method,
                        confidence_score = self._calculate_confidence(sim_score, method),
                    )
                    all_results.append(rr)

            except Exception as e:
                logger.warning(f"⚠️  Retrieval failed for variant '{query_text}': {e}")

        # Rank and deduplicate (notebook Cell 18)
        final = self._rank_and_deduplicate_results(all_results)
        logger.info(f"✅ Retrieved {len(final)} unique parent-context results")
        return final

    def register_parent_chunk(self, parent: ParentChunk) -> None:
        """
        Register a ParentChunk so retrieve_with_parent_child can expand context.
        Called during ingestion (e.g. from orchestrate.py or the API startup).
        """
        self.parent_store[parent.parent_id] = parent
        for child_id in parent.child_chunk_ids:
            self.child_to_parent[child_id] = parent.parent_id

    # ── Private helpers (notebook Cell 18) ────────────────────────────────

    def _calculate_confidence(self, similarity_score: float, method: str) -> float:
        """
        Confidence score combining similarity and retrieval method (notebook Cell 18).

        Method multipliers reflect trustworthiness:
          original   → 1.00 (exact user words)
          hyde       → 0.90 (LLM-generated; slightly less reliable)
          multi_query → 0.85 (paraphrase; further from original)
          bm25       → 0.80 (keyword match; no semantic understanding)
        """
        multipliers = {
            "original":    1.00,
            "hyde":        0.90,
            "multi_query": 0.85,
            "bm25":        0.80,
        }
        m = multipliers.get(method, 0.80)

        # Non-linear scaling: high-similarity results get full multiplier bonus
        if similarity_score > 0.8:
            confidence = similarity_score * m
        elif similarity_score > 0.6:
            confidence = similarity_score * m * 0.9
        else:
            confidence = similarity_score * m * 0.8

        return min(1.0, confidence)

    def _rank_and_deduplicate_results(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Group by parent, keep the best child per parent, then sort by confidence (notebook Cell 18).

        This prevents the same parent context appearing multiple times
        in the final context window when several child chunks matched.
        """
        if not results:
            return []

        # Group by parent_id
        parent_groups: Dict[str, List[RetrievalResult]] = {}
        for r in results:
            pid = r.metadata.get("parent_id", "unknown")
            parent_groups.setdefault(pid, []).append(r)

        # Keep best result per parent
        final: List[RetrievalResult] = []
        for group in parent_groups.values():
            best = max(group, key=lambda x: (x.confidence_score, x.similarity_score))
            final.append(best)

        # Sort by confidence descending
        final.sort(key=lambda x: x.confidence_score, reverse=True)
        return final[: self.retrieval_count]



# ═══════════════════════════════════════════════════════════════════════════
# IMAGE RETRIEVER  (Gap Fix #2)
# ═══════════════════════════════════════════════════════════════════════════

class ImageRetriever:
    """
    Query the image FAISS index using CLIP text embeddings.

    How cross-modal retrieval works
    --------------------------------
    CLIP was trained with a contrastive objective that forces image embeddings
    and text embeddings into the SAME 512-dim space.  A sentence like
    "a printed document or form" produces a text vector that is geometrically
    close to image vectors of scanned paperwork — even though the image was
    never encoded as text.

    At query time we:
      1. Encode the user's query (or its rewritten version) with CLIP's TEXT
         encoder → 512-dim query vector.
      2. Search the image FAISS index (built from CLIP IMAGE vectors) with
         that query vector using inner-product similarity.
      3. Return the top-k image records with their similarity scores,
         captions, geo estimates, and source paths.

    The results can then be merged with text retrieval results in main.py
    to give the LLM both document text AND visual context.
    """

    def __init__(
        self,
        image_faiss_index=None,
        image_chunks:     Optional[List[Dict]] = None,
        top_k:            int = 5,
    ):
        self.image_faiss_index = image_faiss_index
        self.image_chunks      = image_chunks or []
        self.top_k             = top_k

        # CLIP processor + model loaded lazily on first query
        # (avoids loading at import time if image retrieval is disabled)
        self._clip_model     = None
        self._clip_processor = None

        if image_faiss_index is not None:
            logger.info(
                f"🖼️  ImageRetriever ready | "
                f"vectors={image_faiss_index.ntotal}, "
                f"records={len(self.image_chunks)}"
            )
        else:
            logger.info("🖼️  ImageRetriever initialised (no index loaded — image search disabled)")

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Find the most visually relevant images for a text query.

        Args:
            query:  plain-text query (original or rewritten)
            top_k:  number of results; falls back to self.top_k

        Returns:
            List of dicts, each containing:
              image_path    – path to the extracted image file
              source_doc    – PDF stem the image came from
              caption       – CLIP pseudo-caption
              geo_estimate  – StreetCLIP prediction dict
              similarity    – cosine similarity score [0, 1]
              source_type   – always "IMAGE"
        """
        if self.image_faiss_index is None:
            return []

        k = top_k or self.top_k

        try:
            query_vec = self._encode_text(query)          # (1, 512) float32
            distances, indices = self.image_faiss_index.search(query_vec, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.image_chunks):
                    continue
                record = dict(self.image_chunks[idx])     # shallow copy
                record["similarity"]  = float(dist)
                record["source_type"] = "IMAGE"
                results.append(record)

            logger.debug(f"🖼️  Image retrieval: {len(results)} results for '{query[:60]}'")
            return results

        except Exception as e:
            logger.error(f"❌ Image retrieval failed: {e}")
            return []

    def retrieve_by_vector(
        self,
        vector: "np.ndarray",        # (512,) or (1, 512) float32, L2-normalised
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Feature 2 — Search the image FAISS index with a pre-computed vector.

        Called by /query-with-file when the uploaded file is a PNG/image.
        The caller (main.py) already has the CLIP image embedding from
        pipelines.image_pipeline.get_clip_embedding(), so we skip the text
        encoder entirely and search FAISS directly.

        Args:
            vector:  512-dim L2-normalised CLIP image embedding as numpy array.
                     Accepts shape (512,) or (1, 512); reshaped internally.
            top_k:   number of nearest neighbours to return.

        Returns:
            Same structure as retrieve() — list of image chunk dicts with
            an added 'similarity' float field.
        """
        if self.image_faiss_index is None:
            logger.warning("retrieve_by_vector called but image FAISS index not loaded")
            return []

        try:
            import faiss as _faiss

            vec = np.array(vector, dtype="float32").reshape(1, -1)

            # Ensure L2-normalised — safe to call even if already normalised
            _faiss.normalize_L2(vec)

            distances, indices = self.image_faiss_index.search(vec, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.image_chunks):
                    continue
                record = dict(self.image_chunks[idx])   # shallow copy
                record["similarity"]  = float(dist)
                record["source_type"] = "IMAGE"
                results.append(record)

            logger.debug(
                f"🖼️  retrieve_by_vector: {len(results)} results "
                f"(top similarity={results[0]['similarity']:.3f})" if results else
                f"🖼️  retrieve_by_vector: 0 results"
            )
            return results

        except Exception as e:
            logger.error(f"❌ retrieve_by_vector failed: {e}")
            return []

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_clip(self) -> None:
        """Lazy-load CLIP text encoder (only when first query arrives).
        Reads CLIP_MODEL from environment so it stays in sync with the model
        used during indexing in image_pipeline.py — preventing dimension mismatches.
        """
        if self._clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch

                # Must match the model used in image_pipeline.py during indexing.
                # Both read from the same CLIP_MODEL env var so they stay in sync.
                clip_model_name = os.getenv(
                    "CLIP_MODEL", "openai/clip-vit-base-patch32"
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"🔄 Loading CLIP text encoder: {clip_model_name} on {device}…")

                self._clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                self._clip_model     = CLIPModel.from_pretrained(clip_model_name).to(device)
                self._clip_model.eval()
                self._device = device
                logger.info("✅ CLIP text encoder ready")

            except Exception as e:
                logger.error(f"❌ Failed to load CLIP: {e}")
                raise

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into a CLIP text embedding (512-dim, L2-normalised).

        We truncate to 77 tokens — CLIP's hard context window.  Longer queries
        are automatically truncated by the processor with truncation=True.
        """
        import torch
        import torch.nn.functional as F

        self._load_clip()

        inputs = self._clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self._device)

        with torch.no_grad():
            # get_text_features may return a tensor or a structured output;
            # ensure we extract the tensor if needed
            outputs = self._clip_model.get_text_features(**inputs)
            
            # Handle both cases: direct tensor or wrapped output
            if isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                # If it's a BaseModelOutputWithPooling or similar, extract embeddings
                text_features = getattr(outputs, 'pooler_output', outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs)
            
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu().numpy().astype("float32")   # (1, 512)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER  (unchanged from original, used by main.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_context(chunks: List[Dict], max_tokens: int = 3000) -> str:
    """
    Build context string from retrieved chunks with source citations.
    Called by main.py to assemble the LLM prompt context.
    Unchanged from original; kept here for backwards compatibility.
    """
    context_parts = []
    total_tokens  = 0

    for chunk in chunks:
        text     = chunk.get("text", "")
        source   = chunk.get("source_file", "Unknown")
        page     = chunk.get("page", 0)
        doc_type = chunk.get("doc_type", "text")

        citation   = f"\n[Source: {Path(source).stem}, Page {page}, Type: {doc_type}]"
        chunk_text = text + citation

        # Rough token estimate (words / 1.33)
        chunk_tokens = len(text.split()) / 1.33

        if total_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(chunk_text)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    print("retriever.py — import this module; do not run directly.")