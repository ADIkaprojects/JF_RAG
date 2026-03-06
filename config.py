import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
FAISS_DIR       = os.getenv("FAISS_TEXT_INDEX", "./vectorstore/faiss")
BM25_PATH       = os.getenv("BM25_INDEX",       "./vectorstore/bm25.pkl")