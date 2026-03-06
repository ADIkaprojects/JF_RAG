# Files to Upload to Cloudflare R2

These are the **runtime-necessary** artifacts that must be available in production.
Paths are relative to the project root.

## FAISS / Vector Indexes
- vectorstore/faiss/index.bin
- vectorstore/faiss/chunks.json

## BM25 Indexes
- vectorstore/bm25.pkl

## Image Indexes
- vectorstore/image_faiss/index.bin
- vectorstore/image_faiss/chunks.json

## Parent / Auxiliary Indexes
- vectorstore/parent_index.json

## Source Documents (data/)
Upload **all files** under these directories:
- data/sample_document/
- data/sample_image/
- data/csv/
- data/dataset/
- data/text/

These contain the original PDFs, images, CSVs, and derived datasets that the
indexes are built from and are needed for re-ingestion and source-excerpt serving.
