"""
Orchestrate: Full RAG Pipeline Runner
======================================
Upgraded to incorporate the Advanced RAG notebook's patterns:
  - Phase headers match notebook ProductionRAGSystem __init__ log style
  - Uses new DocumentProcessor + IntelligentChunker from upgraded text_pipeline.py
  - Pipeline result dict now includes quality_score stats (notebook Cell 9 style)
  - Deduplication and chunk-count logging unchanged (original project need)
  - All CLI flags unchanged for backwards compatibility
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import Dict, List

# Load .env before anything else so all os.getenv() calls across all pipeline
# modules (image_pipeline.OLLAMA_HOST, embeddings.EMBEDDING_MODEL, etc.)
# see the correct values from the .env file.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate full RAG pipeline: text → image → CSV → embeddings → indexes"
    )

    parser.add_argument("--text",   action="store_true", help="Run text processing pipeline")
    parser.add_argument("--images", action="store_true", help="Run image processing pipeline")
    parser.add_argument("--csv",    action="store_true", help="Run CSV processing pipeline")
    parser.add_argument("--embed",  action="store_true", help="Build embeddings and indexes")
    parser.add_argument("--all",    action="store_true", help="Run all pipelines")
    parser.add_argument(
        "--strategy",
        default="hybrid",
        choices=["fixed", "semantic", "hybrid"],
        help="Chunking strategy for text pipeline (default: hybrid)",
    )
    parser.add_argument("--data-dir",   default="./data",      help="Data directory")
    parser.add_argument("--output-dir", default="./processed", help="Output directory")

    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results    = {}
    start_time = datetime.now()

    # ── Text Pipeline ────────────────────────────────────────────────────────
    if args.all or args.text:
        logger.info("=" * 60)
        logger.info("PHASE 2: TEXT PIPELINE")
        logger.info("   📄 Document Processing: Enterprise-grade PDF + TXT handling")
        logger.info(f"   🧠 Chunking strategy: {args.strategy}")
        logger.info("=" * 60)

        try:
            # Import the upgraded text_pipeline (now uses DocumentProcessor +
            # IntelligentChunker from notebook Cell 9/11)
            from pipelines.text_pipeline import process_document_folder

            chunks_dir = output_dir / "chunks"
            all_chunks = []

            # Process sample_document folder (PDFs)
            text_dir_1 = data_dir / "sample_document"
            if text_dir_1.exists():
                logger.info(f"🔄 Processing {text_dir_1}…")
                chunks = process_document_folder(
                    str(text_dir_1), str(chunks_dir), strategy=args.strategy
                )
                all_chunks.extend(chunks)
                logger.info(f"   {len(chunks)} chunks from sample_document")

            # Process data/text folder (.txt files)
            text_dir_2 = data_dir / "text"
            if text_dir_2.exists():
                logger.info(f"🔄 Processing {text_dir_2}…")
                chunks = process_document_folder(
                    str(text_dir_2), str(chunks_dir), strategy=args.strategy
                )
                all_chunks.extend(chunks)
                logger.info(f"   {len(chunks)} chunks from data/text")

            # Notebook-style quality summary
            if all_chunks:
                # DocumentChunk objects have metadata.semantic_score
                try:
                    scores = [c.metadata.semantic_score for c in all_chunks]
                    avg_quality = sum(scores) / len(scores) if scores else 0.0
                    logger.info(
                        f"✅ Text pipeline: {len(all_chunks)} total chunks | "
                        f"avg semantic_score={avg_quality:.2f}"
                    )
                    results["text_pipeline"] = {
                        "status":           "success",
                        "chunks_processed": len(all_chunks),
                        "avg_quality":      round(avg_quality, 4),
                        "output_dir":       str(chunks_dir),
                        "strategy":         args.strategy,
                    }
                except AttributeError:
                    # Fallback if chunks are plain dicts (older serialised format)
                    logger.info(f"✅ Text pipeline: {len(all_chunks)} total chunks")
                    results["text_pipeline"] = {
                        "status":           "success",
                        "chunks_processed": len(all_chunks),
                        "output_dir":       str(chunks_dir),
                    }
            else:
                logger.warning("⚠️  No chunks produced by text pipeline")
                results["text_pipeline"] = {
                    "status": "no_data",
                    "chunks_processed": 0,
                    "output_dir": str(chunks_dir),
                }

        except Exception as e:
            logger.error(f"❌ Text pipeline failed: {e}")
            results["text_pipeline"] = {"status": "failed", "error": str(e)}

    # ── Image Pipeline ───────────────────────────────────────────────────────
    if args.all or args.images:
        logger.info("=" * 60)
        logger.info("PHASE 3: IMAGE PIPELINE")
        logger.info("=" * 60)

        try:
            from pipelines.image_pipeline import process_image_folder

            image_dir    = data_dir / "sample_image"
            extract_dir  = output_dir / "images"
            metadata_dir = output_dir / "image_meta"

            metadata = process_image_folder(
                str(image_dir), str(extract_dir), str(metadata_dir)
            )
            results["image_pipeline"] = {
                "status":           "success",
                "images_processed": len(metadata),
                "extract_dir":      str(extract_dir),
                "metadata_dir":     str(metadata_dir),
            }
            logger.info(f"✅ Image pipeline: {len(metadata)} images processed")

            # ── Build image FAISS index immediately after extraction ──────
            # image_metadata.json (with 512-dim CLIP embeddings) was just
            # written by process_image_folder.  We build the image FAISS
            # index here so main.py can load it on startup without needing
            # a separate --embed-images flag.
            full_meta_path = metadata_dir / "image_metadata.json"
            if full_meta_path.exists() and metadata:
                logger.info("🔧 Building image FAISS index…")
                try:
                    from retrieval.embeddings import build_image_indexes
                    vectorstore_dir = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
                    img_index_result = build_image_indexes(
                        str(full_meta_path),
                        str(vectorstore_dir),
                    )
                    results["image_pipeline"]["image_faiss"] = img_index_result
                    logger.info(
                        f"✅ Image FAISS index built: "
                        f"{img_index_result['num_images']} vectors, "
                        f"{img_index_result['embedding_dim']}-dim → "
                        f"{img_index_result['image_faiss_dir']}"
                    )
                except Exception as ie:
                    logger.error(f"❌ Image FAISS build failed: {ie}")
                    results["image_pipeline"]["image_faiss"] = {"status": "failed", "error": str(ie)}
            else:
                logger.info("   ℹ️  No image metadata to index (empty dataset or metadata missing)")

        except Exception as e:
            logger.error(f"❌ Image pipeline failed: {e}")
            results["image_pipeline"] = {"status": "failed", "error": str(e)}

    # ── CSV Pipeline ─────────────────────────────────────────────────────────
    if args.all or args.csv:
        logger.info("=" * 60)
        logger.info("PHASE 4: CSV PIPELINE")
        logger.info("=" * 60)

        try:
            from pipelines.csv_pipeline import process_csv_folder

            csv_dir     = data_dir / "csv"
            csv_out_dir = output_dir / "chunks"

            records = []
            if csv_dir.exists():
                # process_csv_folder already uses glob("**/*.csv") — one call covers
                # all CSVs in csv_dir and every subdirectory (e.g. csv/file/).
                batch = process_csv_folder(str(csv_dir), str(csv_out_dir))
                records.extend(batch)
                logger.info(f"   Processed CSVs from {csv_dir.name} (recursive)")

            # Notebook-style quality summary
            if records:
                avg_quality = sum(r.quality_score for r in records) / len(records)
                logger.info(
                    f"✅ CSV pipeline: {len(records)} records | "
                    f"avg quality={avg_quality:.2f}"
                )
                results["csv_pipeline"] = {
                    "status":            "success",
                    "records_processed": len(records),
                    "avg_quality":       round(avg_quality, 4),
                    "output_dir":        str(csv_out_dir),
                }
            else:
                results["csv_pipeline"] = {
                    "status":            "no_data",
                    "records_processed": 0,
                    "output_dir":        str(csv_out_dir),
                }

        except Exception as e:
            logger.error(f"❌ CSV pipeline failed: {e}")
            results["csv_pipeline"] = {"status": "failed", "error": str(e)}

    # ── Embedding & Indexing ─────────────────────────────────────────────────
    if args.all or args.embed:
        logger.info("=" * 60)
        logger.info("PHASE 5: EMBEDDING & INDEXING")
        logger.info("   🗄️  Vector Storage: FAISS + BM25 hybrid indexes")
        logger.info("=" * 60)

        try:
            from retrieval.embeddings import build_text_indexes

            chunks_dir      = output_dir / "chunks"
            vectorstore_dir = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
            combined_path   = chunks_dir / "_all_chunks.json"

            # Collect all per-file chunk JSONs (skip the combined file)
            chunks_files = [
                f for f in chunks_dir.glob("*.json")
                if f.name != "_all_chunks.json"
            ]

            if not chunks_files:
                logger.warning("⚠️  No chunk files found to embed")
                results["embedding"] = {"status": "skipped", "reason": "No chunk files"}
            else:
                logger.info(f"   Found {len(chunks_files)} chunk files to combine")

                # ── Pass 1: ID-based dedup (original) ────────────────
                # Removes chunks whose chunk_id / record_id has been seen
                # before — handles re-runs and overlapping input folders.
                seen_ids:   set        = set()
                all_chunks: List[Dict] = []

                for chunks_file in chunks_files:
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        file_chunks = json.load(f)

                    before = len(all_chunks)
                    for chunk in file_chunks:
                        cid = chunk.get("chunk_id") or chunk.get("record_id")
                        if cid and cid in seen_ids:
                            continue
                        if cid:
                            seen_ids.add(cid)
                        all_chunks.append(chunk)

                    added = len(all_chunks) - before
                    logger.info(
                        f"   {chunks_file.name}: {len(file_chunks)} chunks, {added} new (ID deduped)"
                    )

                logger.info(f"   After ID dedup: {len(all_chunks)} chunks")

                # ── Pass 2: Content-hash dedup (new) ─────────────────────
                # Catches semantically duplicate chunks that survived Pass 1
                # because they came from different source files (e.g. the
                # same email forwarded as multiple PDFs) or because the
                # page-level / intra-document dedup in text_pipeline.py
                # couldn't see across file boundaries.
                #
                # Fingerprint: MD5 of the first 200 chars of the chunk text,
                # lowercased and whitespace-collapsed.  200 chars is enough
                # to uniquely identify any real chunk while being forgiving
                # of trailing-whitespace or minor OCR differences at the end.
                # Chunks shorter than 30 chars are never hashed (they might
                # be meaningful single-word labels).
                seen_content_hashes: set = set()
                content_deduped:     List[Dict] = []

                for chunk in all_chunks:
                    text = chunk.get("text", "")
                    if len(text) < 30:
                        # Too short to fingerprint reliably — always keep
                        content_deduped.append(chunk)
                        continue

                    fingerprint = hashlib.md5(
                        " ".join(text[:200].lower().split()).encode()
                    ).hexdigest()

                    if fingerprint in seen_content_hashes:
                        continue   # cross-file duplicate — discard
                    seen_content_hashes.add(fingerprint)
                    content_deduped.append(chunk)

                content_dropped = len(all_chunks) - len(content_deduped)
                if content_dropped:
                    logger.info(
                        f"   🔁 Cross-file content dedup: removed {content_dropped} "
                        f"duplicate chunk(s) — {len(content_deduped)} remain"
                    )
                all_chunks = content_deduped

                logger.info(f"   Total unique chunks to embed: {len(all_chunks)}")

                # Save deduplicated combined file
                with open(combined_path, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                logger.info(f"   💾 Saved deduplicated chunks → {combined_path}")

                # Build indexes (build_text_indexes now returns embed_stats too)
                index_result = build_text_indexes(
                    str(combined_path),
                    str(vectorstore_dir),
                    embedding_model_name=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
                )
                results["embedding"] = {
                    "status": "success",
                    **index_result,
                }
                logger.info(
                    f"✅ Embedding: {index_result['num_chunks']} chunks indexed | "
                    f"model={index_result['model']} | device={index_result['device']}"
                )

                # After building indexes, sync them to R2 (no-op in development)
                try:
                    from api.main import sync_indexes_to_r2  # lazy import to avoid heavy startup cost

                    sync_indexes_to_r2()
                    logger.info("   ☁️  Indexes synced to Cloudflare R2 (if in production).")
                except Exception as sync_err:  # noqa: BLE001
                    logger.warning(f"⚠️  R2 sync after embedding failed: {sync_err}")

        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            results["embedding"] = {"status": "failed", "error": str(e)}

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    elapsed = (datetime.now() - start_time).total_seconds()

    for pipeline, result in results.items():
        status = result.get("status", "unknown")
        emoji  = "✅" if status == "success" else ("⚠️ " if status in ("skipped", "no_data") else "❌")
        logger.info(f"   {emoji} {pipeline}: {status}")

    logger.info(f"   ⏱  Total time: {elapsed:.2f} seconds")

    # Persist pipeline results (unchanged from original)
    results_file = output_dir / "pipeline_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp":       start_time.isoformat(),
                "elapsed_seconds": elapsed,
                "results":         results,
            },
            f,
            indent=2,
        )

    logger.info(f"   📄 Results saved → {results_file}")



if __name__ == "__main__":
    main()