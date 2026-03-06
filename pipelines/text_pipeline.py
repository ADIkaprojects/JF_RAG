"""
Text Pipeline: PDF / TXT → Chunked Documents with Metadata
============================================================
Upgraded to incorporate the Advanced RAG notebook's patterns:
  - DocumentMetadata / ProcessedDocument dataclasses for richer tracking
  - IntelligentChunker with fixed / semantic / hybrid strategies (notebook Cell 11)
  - Token-aware chunking via tiktoken (kept from original)
  - Quality scoring per chunk and per document (notebook Cell 9 _assess_text_quality)
  - OCR fallback with PyMuPDF + pytesseract (kept from original project need)
  - Email / transcript boundary detection (kept from original project need)
  - process_document_folder entry-point that handles both PDF and .txt files
"""

import fitz           # PyMuPDF  – primary PDF extractor
import pytesseract    # OCR fallback for scanned PDFs
import re
import json
import uuid
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from PIL import Image
import tiktoken
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES  (notebook Cell 9 style – richer than original TextChunk)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentMetadata:
    """
    Comprehensive metadata for document tracking (notebook Cell 9).
    Replaces the bare dict that was used in the original pipeline.
    """
    document_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    filename:             str   = ""
    total_pages:          int   = 0
    total_characters:     int   = 0
    processing_timestamp: str   = field(default_factory=lambda: datetime.now().isoformat())
    language:             str   = "en"
    quality_score:        float = 0.0
    # 'native' | 'mixed' | 'scanned' | 'txt' – kept from original project
    extraction_method:    str   = "native"


@dataclass
class ProcessedDocument:
    """
    Container for a fully processed document (notebook Cell 9).
    page_contents mirrors the notebook's per-page quality dict.
    """
    content:       str
    metadata:      DocumentMetadata
    page_contents: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """
    Rich per-chunk metadata (notebook Cell 11).
    Supersedes the original bare TextChunk dataclass.
    """
    chunk_id:            str        = field(default_factory=lambda: str(uuid.uuid4()))
    parent_document_id:  str        = ""
    chunk_index:         int        = 0
    start_char:          int        = 0
    end_char:            int        = 0
    token_count:         int        = 0
    page_numbers:        List[int]  = field(default_factory=list)
    # 'content' | 'header' | 'footer' | 'table' (notebook) +
    # 'email'  | 'transcript' (project-specific)
    chunk_type:          str        = "content"
    semantic_score:      float      = 0.0
    # Extra project fields kept from original
    source_file:         str        = ""
    doc_type:            str        = "native"
    ocr_quality:         float      = 1.0
    chunking_strategy:   str        = "hybrid"
    created_at:          str        = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DocumentChunk:
    """
    Individual document chunk (notebook Cell 11).
    Replaces original TextChunk; embedding field kept for downstream use.
    """
    content:  str
    metadata: ChunkMetadata


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR  (notebook Cell 9 style, keeps original OCR fallback)
# ═══════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """
    Production-grade document processor (notebook Cell 9 pattern).

    Changes vs original text_pipeline.py:
      • Returns ProcessedDocument instead of a raw list-of-page-dicts
      • _assess_text_quality / _calculate_document_quality from notebook
      • _clean_text upgraded to notebook version (artifact removal, quote fix)
      • OCR fallback via PyMuPDF + pytesseract (kept from original)
    """

    def __init__(self, min_chars_per_page: int = 100):
        self.min_chars_per_page = min_chars_per_page

    # ── Public entry-point ─────────────────────────────────────────────────

    def load_pdf(self, pdf_path: str) -> ProcessedDocument:
        """
        Load and process a PDF file with quality assessment (notebook Cell 9).
        Automatically falls back to OCR if the PDF is scanned.
        """
        logger.info(f"🔄 Processing PDF: {pdf_path}")
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Classify so we know whether OCR is needed (kept from original)
        doc_type = self._classify_pdf(pdf_path)
        logger.info(f"   Classified as: {doc_type}")

        try:
            doc = fitz.open(pdf_path)
            metadata = DocumentMetadata(
                filename=Path(pdf_path).name,
                total_pages=len(doc),
                extraction_method=doc_type,
            )

            page_contents = []
            full_text = ""

            # ── Intra-document duplicate-page guard ──────────────────────
            # Scanned PDFs like this one (EFTA00037195) often embed the same
            # content twice: once as a thin native-text layer and once as the
            # OCR result of the scanned image on the same page.  We fingerprint
            # each page's cleaned text and skip any page whose fingerprint has
            # already been seen within this document.
            #
            # Fingerprint: MD5 of the first 300 chars, lower-cased and
            # whitespace-collapsed.  This is robust to trivial OCR variance
            # (trailing spaces, minor char swaps) while still catching true
            # duplicates.  Pages shorter than 50 chars are never skipped —
            # they could be meaningful single-line headers.
            seen_page_hashes: Set[str] = set()

            for page_num, page in enumerate(doc):
                try:
                    # Extract raw text; fall back to OCR for scanned pages
                    if doc_type == "scanned":
                        raw_text = self._ocr_page(page)
                        ocr_quality = self._estimate_ocr_quality(raw_text, page)
                    else:
                        raw_text = page.get_text("text")
                        ocr_quality = 1.0

                    cleaned_text = self._clean_text(raw_text)

                    # ── Skip duplicate pages ──────────────────────────────
                    page_fingerprint = hashlib.md5(
                        " ".join(cleaned_text[:300].lower().split()).encode()
                    ).hexdigest()

                    if page_fingerprint in seen_page_hashes and len(cleaned_text) > 50:
                        logger.info(
                            f"   ⚠️  Page {page_num + 1}: duplicate of earlier page "
                            f"— skipped (double-render guard)"
                        )
                        continue   # do NOT add to page_contents or full_text
                    seen_page_hashes.add(page_fingerprint)
                    # ─────────────────────────────────────────────────────

                    page_info = {
                        "page_number": page_num + 1,
                        "raw_text":    raw_text,
                        "cleaned_text": cleaned_text,
                        "char_count":   len(cleaned_text),
                        # Notebook-style quality score per page
                        "quality_score": self._assess_text_quality(cleaned_text),
                        "ocr_quality":   ocr_quality,
                    }
                    page_contents.append(page_info)
                    full_text += cleaned_text + "\n\n"

                    logger.debug(f"   Page {page_num + 1}: {len(cleaned_text)} chars, "
                                 f"quality={page_info['quality_score']:.2f}")

                except Exception as e:
                    logger.warning(f"⚠️  Error on page {page_num + 1}: {e}")
                    page_contents.append({
                        "page_number": page_num + 1,
                        "raw_text": "", "cleaned_text": "",
                        "char_count": 0, "quality_score": 0.0,
                        "ocr_quality": 0.0, "error": str(e),
                    })

            doc.close()

            # Finalise metadata (notebook Cell 9 pattern)
            metadata.total_characters = len(full_text)
            metadata.quality_score    = self._calculate_document_quality(page_contents)

            logger.info(f"✅ PDF processed – pages={metadata.total_pages}, "
                        f"chars={metadata.total_characters:,}, "
                        f"quality={metadata.quality_score:.2f}")

            return ProcessedDocument(
                content=full_text.strip(),
                metadata=metadata,
                page_contents=page_contents,
            )

        except Exception as e:
            logger.error(f"❌ Failed to process PDF {pdf_path}: {e}")
            raise

    def load_txt(self, txt_path: str) -> ProcessedDocument:
        """Load a plain-text file and wrap it in a ProcessedDocument."""
        logger.info(f"🔄 Processing TXT: {txt_path}")
        try:
            text = Path(txt_path).read_text(encoding="utf-8")
            logger.info(f"   Read {len(text)} characters")

            cleaned = self._clean_text(text)
            quality = self._assess_text_quality(cleaned)

            metadata = DocumentMetadata(
                filename=Path(txt_path).name,
                total_pages=1,
                total_characters=len(cleaned),
                quality_score=quality,
                extraction_method="txt",
            )

            page_contents = [{
                "page_number": 1,
                "raw_text": text,
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
                "quality_score": quality,
                "ocr_quality": 1.0,
            }]

            return ProcessedDocument(
                content=cleaned,
                metadata=metadata,
                page_contents=page_contents,
            )

        except Exception as e:
            logger.error(f"❌ Failed to load TXT {txt_path}: {e}")
            return ProcessedDocument(
                content="",
                metadata=DocumentMetadata(filename=Path(txt_path).name),
                page_contents=[],
            )

    # ── Private helpers ────────────────────────────────────────────────────

    def _classify_pdf(self, pdf_path: str) -> str:
        """
        Classify PDF as native / mixed / scanned.

        Extended: after the char-ratio check, a noise-quality check catches
        scanned PDFs with a garbage embedded text layer (e.g. a photo of a
        table where PyMuPDF extracts 2000+ chars of garbled OCR noise).
        If text looks noisy (low alpha ratio or very long avg word length)
        the classification is downgraded native -> mixed.
        """
        try:
            doc        = fitz.open(pdf_path)
            pages_text = [p.get_text("text") for p in doc]
            doc.close()

            total_text = sum(len(t) for t in pages_text)
            n          = max(len(pages_text), 1)
            ratio      = total_text / n

            if ratio <= 20:
                return "scanned"
            if ratio <= 100:
                return "mixed"

            # ratio > 100 looks native — verify text quality
            sample = " ".join(pages_text[0].split())[:500]
            if sample:
                alpha_chars  = sum(1 for c in sample if c.isalpha())
                alpha_ratio  = alpha_chars / max(len(sample), 1)
                words        = sample.split()
                avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

                # Garbled OCR tokens inflate avg_word_len; low alpha = noise
                if alpha_ratio < 0.55 or avg_word_len > 12:
                    logger.info(
                        f"   ⚠️  Native text looks noisy "
                        f"(alpha_ratio={alpha_ratio:.2f}, avg_word_len={avg_word_len:.1f})"
                        f" — reclassifying as mixed"
                    )
                    return "mixed"

            return "native"

        except Exception as e:
            logger.error(f"Classification error for {pdf_path}: {e}")
            return "unknown"

    def _ocr_page(self, page, dpi: int = 300) -> str:
        """Run Tesseract OCR on a single PDF page (kept from original)."""
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, config="--psm 6")

    def _estimate_ocr_quality(self, text: str, page) -> float:
        """Heuristic OCR quality score (kept from original)."""
        pix = page.get_pixmap()
        area = max(pix.width * pix.height / 10000, 1)
        return min(1.0, len(text.split()) / area)

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalise extracted text (notebook Cell 9 version).
        Keeps original OCR-specific cleaning on top.
        """
        if not text:
            return ""

        # Remove control characters (original)
        text = re.sub(r"[\x00-\x08\x0B-\x1F]", "", text)

        # Fix OCR hyphenation (original)
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Collapse excessive whitespace — but preserve paragraph separators (\n\n)
        # so that _semantic_chunking() can still split on paragraph boundaries.
        # We collapse runs of spaces/tabs on each line, then normalise multiple
        # blank lines down to a single \n\n.
        lines = [" ".join(line.split()) for line in text.split("\n")]
        text  = "\n".join(lines)
        text  = re.sub(r"\n{3,}", "\n\n", text)   # max 1 blank line between paragraphs

        # Remove common PDF artifacts – null / replacement / bullet (notebook)
        for artifact in ["\x00", "\ufffd", "\u2022"]:
            text = text.replace(artifact, "")

        # Normalise quotes and dashes (notebook)
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Flag redactions (original)
        text = re.sub(r"(\u2588+|_{4,}|\[+REDACTED\]+)", "[REDACTED]", text)

        return text.strip()

    def _assess_text_quality(self, text: str) -> float:
        """
        Per-page text quality score 0–1 (notebook Cell 9).
        Kept identical to notebook implementation.
        """
        if not text:
            return 0.0

        char_count  = len(text)
        word_count  = len(text.split())

        # Penalise very short text
        length_score = min(char_count / self.min_chars_per_page, 1.0)

        # Reasonable average word length (3–8 chars → good)
        avg_word_len = char_count / max(word_count, 1)
        word_ratio_score = 1.0 if 3 <= avg_word_len <= 8 else 0.5

        # Penalise excessive special characters (OCR noise indicator)
        special = sum(1 for c in text if not c.isalnum() and c not in " .,!?;:-\"\n")
        special_char_score = 1.0 - min((special / max(len(text), 1)) * 2, 0.5)

        return (length_score + word_ratio_score + special_char_score) / 3

    def _calculate_document_quality(self, page_contents: List[Dict]) -> float:
        """Average quality score across all pages (notebook Cell 9)."""
        if not page_contents:
            return 0.0
        scores = [p.get("quality_score", 0.0) for p in page_contents]
        return sum(scores) / len(scores)


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENT CHUNKER  (notebook Cell 11 – full implementation)
# ═══════════════════════════════════════════════════════════════════════════

# Email boundary pattern kept from original for project-specific need
_EMAIL_BOUNDARY = re.compile(r"(?:^|\n)(?:From:|Sent:|On .+ wrote:)", re.MULTILINE)


class IntelligentChunker:
    """
    Advanced chunking with fixed / semantic / hybrid strategies (notebook Cell 11).

    Key changes vs original text_pipeline.py:
      • Unified class instead of scattered module-level functions
      • _fixed_size_chunking, _semantic_chunking, _hybrid_chunking from notebook
      • _post_process_chunks adds semantic_score and page_numbers (notebook)
      • Email / transcript routing kept for project-specific docs
      • Returns List[DocumentChunk] (new dataclass) instead of List[TextChunk]
    """

    def __init__(
        self,
        chunk_size:     int = 1000,
        chunk_overlap:  int = 200,
        min_chunk_size: int = 10,
        max_chunk_size: int = 2000,
        strategy:       str = "hybrid",   # "fixed" | "semantic" | "hybrid"
    ):
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy       = strategy

        # tiktoken kept from original for token-aware splitting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"🔧 IntelligentChunker: strategy={strategy}, "
                    f"size={chunk_size}, overlap={chunk_overlap}")

    # ── Public entry-point ─────────────────────────────────────────────────

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Chunk a ProcessedDocument using the configured strategy (notebook Cell 11).
        Also handles email-thread and transcript routing (original project need).
        """
        logger.info(f"🔄 Chunking: {document.metadata.filename} | strategy={self.strategy}")

        # Project-specific routing: check for email threads or transcripts
        # before falling back to the notebook strategies
        text_sample = document.content[:5000]
        is_email      = bool(_EMAIL_BOUNDARY.search(text_sample))
        is_transcript = bool(re.search(r"^\s*Q:", document.content, re.MULTILINE))

        if is_email:
            # Email threads need boundary-aware splitting (original project need)
            chunks = self._email_chunking(document)
        elif is_transcript:
            chunks = self._semantic_chunking(document)
        elif self.strategy == "fixed":
            chunks = self._fixed_size_chunking(document)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunking(document)
        elif self.strategy == "hybrid":
            chunks = self._hybrid_chunking(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        # Post-processing: index, quality score, page attribution (notebook Cell 11)
        chunks = self._post_process_chunks(chunks, document)

        if chunks:
            lengths = [len(c.content) for c in chunks]
            logger.info(f"✅ {len(chunks)} chunks created | "
                        f"avg={np.mean(lengths):.0f} chars | "
                        f"range={min(lengths)}–{max(lengths)} chars")
        else:
            logger.warning(f"⚠️  0 chunks created from {document.metadata.filename} "
                           f"(document too short or below min_chunk_size={self.min_chunk_size})")
        return chunks

    # ── Strategy implementations (notebook Cell 11) ────────────────────────

    def _fixed_size_chunking(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Fixed-size chunks with sentence-boundary awareness (notebook Cell 11)."""
        text   = document.content
        chunks = []
        start  = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a sentence boundary near the target end
            if end < len(text):
                search_start = max(end - 100, start)
                sentences    = text[search_start:end + 50].split(". ")
                if len(sentences) > 1:
                    end = search_start + len(". ".join(sentences[:-1])) + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                meta = ChunkMetadata(
                    parent_document_id = document.metadata.document_id,
                    chunk_index        = len(chunks),
                    start_char         = start,
                    end_char           = end,
                    token_count        = max(1, len(chunk_text) // 4),  # fast char-based estimate
                    source_file        = document.metadata.filename,
                    doc_type           = document.metadata.extraction_method,
                    chunking_strategy  = "fixed",
                )
                chunks.append(DocumentChunk(content=chunk_text, metadata=meta))

            # Overlap: move start back by chunk_overlap (notebook Cell 11)
            # Guard: if overlap would push start back to where we already were,
            # advance past end to prevent an infinite loop
            next_start = end - self.chunk_overlap
            if next_start <= start or next_start < 0:
                start = end
            else:
                start = next_start

        return chunks

    def _semantic_chunking(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Paragraph-boundary semantic chunking (notebook Cell 11).
        Accumulates paragraphs until max_chunk_size would be exceeded.
        """
        paragraphs = document.content.split("\n\n")
        chunks     = []
        current    = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            potential = current + "\n\n" + para if current else para

            if len(potential) <= self.max_chunk_size:
                current = potential
            else:
                if len(current) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk(current, document, len(chunks), current_start)
                    )
                    current_start += len(current) + 2   # +2 for the \n\n separator
                current = para

        # Flush remaining text
        if len(current) >= self.min_chunk_size:
            chunks.append(
                self._create_chunk(current, document, len(chunks), current_start)
            )

        return chunks

    def _hybrid_chunking(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Hybrid: semantic boundaries + size constraints (notebook Cell 11).
        Any semantic chunk exceeding max_chunk_size is split further.
        """
        semantic_chunks = self._semantic_chunking(document)
        final_chunks    = []

        for chunk in semantic_chunks:
            if len(chunk.content) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split oversized semantic chunk into fixed-size sub-chunks
                sub_chunks = self._split_large_chunk(chunk, document)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _email_chunking(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Split email threads on From:/Sent: boundaries (original project need).
        Long individual messages fall back to _semantic_chunking.
        """
        messages = _EMAIL_BOUNDARY.split(document.content)
        chunks   = []

        for msg in messages:
            msg = msg.strip()
            if not msg:
                continue

            if len(msg) // 4 <= self.chunk_size:  # fast char-based estimate
                chunks.append(
                    self._create_chunk(msg, document, len(chunks), 0)
                )
            else:
                # Create a temporary doc so _semantic_chunking can be reused
                tmp_doc = ProcessedDocument(
                    content=msg,
                    metadata=DocumentMetadata(
                        document_id=document.metadata.document_id,
                        filename=document.metadata.filename,
                        extraction_method="email",
                    ),
                )
                sub_chunks = self._semantic_chunking(tmp_doc)
                for sc in sub_chunks:
                    sc.metadata.chunking_strategy = "email_boundary"
                chunks.extend(sub_chunks)

        return chunks

    # ── Internal helpers ───────────────────────────────────────────────────

    def _split_large_chunk(
        self,
        chunk:    DocumentChunk,
        document: ProcessedDocument,
    ) -> List[DocumentChunk]:
        """Break an oversized chunk using sentence-boundary splitting (notebook Cell 11)."""
        text       = chunk.content
        sub_chunks = []
        start      = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Walk back to nearest sentence end
            if end < len(text):
                for i in range(end, max(end - 200, start), -1):
                    if text[i] in ".!?":
                        end = i + 1
                        break

            sub_text = text[start:end].strip()
            if len(sub_text) >= self.min_chunk_size:
                meta = ChunkMetadata(
                    parent_document_id = document.metadata.document_id,
                    chunk_index        = chunk.metadata.chunk_index + len(sub_chunks),
                    start_char         = chunk.metadata.start_char + start,
                    end_char           = chunk.metadata.start_char + end,
                    token_count        = max(1, len(sub_text) // 4),  # fast char-based estimate
                    source_file        = document.metadata.filename,
                    doc_type           = document.metadata.extraction_method,
                    chunking_strategy  = "hybrid_split",
                )
                sub_chunks.append(DocumentChunk(content=sub_text, metadata=meta))

            # Guard against infinite loop: if overlap pushes start back
            # to or before where we already were, advance past end instead
            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start

        return sub_chunks

    def _create_chunk(
        self,
        text:     str,
        document: ProcessedDocument,
        index:    int,
        start_pos: int,
    ) -> DocumentChunk:
        """Helper: create a DocumentChunk with full metadata (notebook Cell 11)."""
        meta = ChunkMetadata(
            parent_document_id = document.metadata.document_id,
            chunk_index        = index,
            start_char         = start_pos,
            end_char           = start_pos + len(text),
            token_count        = max(1, len(text) // 4),  # fast char-based estimate; avoids tiktoken hang on noisy OCR text
            source_file        = document.metadata.filename,
            doc_type           = document.metadata.extraction_method,
            chunking_strategy  = self.strategy,
        )
        return DocumentChunk(content=text, metadata=meta)

    def _post_process_chunks(
        self,
        chunks:   List[DocumentChunk],
        document: ProcessedDocument,
    ) -> List[DocumentChunk]:
        """
        Re-index, score, and attribute page numbers (notebook Cell 11).
        Also preserves OCR quality from page_contents (original project need).
        """
        for i, chunk in enumerate(chunks):
            chunk.metadata.chunk_index    = i
            chunk.metadata.semantic_score = self._calculate_chunk_quality(chunk)

            if document.page_contents:
                chunk.metadata.page_numbers = self._get_page_numbers(chunk, document)

                # Propagate OCR quality from the page(s) this chunk spans
                if chunk.metadata.page_numbers:
                    pg = chunk.metadata.page_numbers[0]
                    pg_info = next(
                        (p for p in document.page_contents if p["page_number"] == pg), None
                    )
                    if pg_info:
                        chunk.metadata.ocr_quality = pg_info.get("ocr_quality", 1.0)

        return chunks

    def _calculate_chunk_quality(self, chunk: DocumentChunk) -> float:
        """
        Quality score 0–1 for a chunk (notebook Cell 11).
        Combines length adequacy, sentence completeness, and word density.
        """
        content = chunk.content
        length  = len(content)

        # Closeness to target chunk size
        length_score = 1.0 - abs(length - self.chunk_size) / self.chunk_size
        length_score = max(0.0, min(1.0, length_score))

        # Does it end on a sentence boundary?
        completeness_score = 1.0 if content.rstrip().endswith((".", "!", "?")) else 0.7

        # Word density (words per char)
        words         = len(content.split())
        density_score = min(words / max(length / 5, 1), 1.0) if length > 0 else 0.0

        return (length_score + completeness_score + density_score) / 3

    def _get_page_numbers(
        self,
        chunk:    DocumentChunk,
        document: ProcessedDocument,
    ) -> List[int]:
        """
        Determine which pages a chunk spans (notebook Cell 11 + original).
        Uses character offset tracking to find overlapping pages.
        """
        pages       = []
        current_pos = 0

        for page_info in document.page_contents:
            page_text  = page_info.get("cleaned_text", "")
            page_start = current_pos
            page_end   = current_pos + len(page_text)

            if (chunk.metadata.start_char < page_end and
                    chunk.metadata.end_char > page_start):
                pages.append(page_info["page_number"])

            current_pos = page_end + 2   # +2 for the \n\n separator

        return pages



# ═══════════════════════════════════════════════════════════════════════════
# INTRA-DOCUMENT CHUNK DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def _dedup_chunks(chunks: List[DocumentChunk], similarity_threshold: float = 0.85) -> List[DocumentChunk]:
    """
    Remove near-duplicate chunks produced from a single document.

    Why this is needed
    ------------------
    Scanned PDFs like EFTA00037195 can produce duplicate text in two ways:
      1. The page-level guard in load_pdf catches *exact* page duplicates.
      2. But if the native-text layer and the OCR layer differ slightly
         (different line breaks, minor char substitutions), the page guard
         misses them — and after chunking we end up with two chunks whose
         text is ~90 % identical.

    Algorithm
    ---------
    For each chunk we compute a "bag-of-words" fingerprint: the frozenset of
    unique lowercased words in the chunk.  Two chunks are considered duplicates
    if their Jaccard similarity exceeds `similarity_threshold`.

      Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    We keep the chunk with the higher semantic_score (quality) and discard the
    other.  The first chunk always survives (nothing to compare against).

    Complexity: O(n²) over chunks within one document.  Documents rarely
    produce more than a few hundred chunks, so this is acceptable.  For very
    large documents (1000+ chunks) consider switching to MinHash LSH.

    Args:
        chunks:               list of DocumentChunk from a single document
        similarity_threshold: Jaccard threshold above which two chunks are
                              considered duplicates (default 0.85)

    Returns:
        Deduplicated list; chunk_index values are re-assigned sequentially.
    """
    if not chunks:
        return chunks

    # Build word-set fingerprints once
    word_sets: List[frozenset] = [
        frozenset(c.content.lower().split()) for c in chunks
    ]

    keep: List[bool] = [True] * len(chunks)

    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            a, b = word_sets[i], word_sets[j]
            union = len(a | b)
            if union == 0:
                continue
            jaccard = len(a & b) / union
            if jaccard >= similarity_threshold:
                # Keep the higher-quality chunk; discard the other
                if chunks[i].metadata.semantic_score >= chunks[j].metadata.semantic_score:
                    keep[j] = False
                else:
                    keep[i] = False
                    break   # chunk i is now discarded; stop inner loop

    deduped = [c for c, k in zip(chunks, keep) if k]

    dropped = len(chunks) - len(deduped)
    if dropped:
        logger.info(
            f"   🔁 Intra-document dedup: removed {dropped} near-duplicate chunk(s) "
            f"(Jaccard ≥ {similarity_threshold}) — {len(deduped)} remain"
        )

    # Re-assign chunk_index so it stays sequential
    for idx, chunk in enumerate(deduped):
        chunk.metadata.chunk_index = idx

    return deduped


# ═══════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPER  (matches original JSON format for orchestrate.py)
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_to_dict(chunk: DocumentChunk) -> Dict:
    """
    Serialise a DocumentChunk to the flat dict expected by embeddings.py.
    Maintains backwards-compatibility with the original chunk JSON schema.
    """
    m = chunk.metadata
    return {
        # Fields consumed by embeddings.py / retriever.py
        "chunk_id":    m.chunk_id,
        "text":        chunk.content,
        "source_file": m.source_file,
        "page":        m.page_numbers[0] if m.page_numbers else 0,
        "doc_type":    m.doc_type,
        "token_count": m.token_count,
        "ocr_quality": m.ocr_quality,
        # Richer fields added from notebook
        "parent_document_id": m.parent_document_id,
        "chunk_index":        m.chunk_index,
        "start_char":         m.start_char,
        "end_char":           m.end_char,
        "chunk_type":         m.chunk_type,
        "semantic_score":     m.semantic_score,
        "chunking_strategy":  m.chunking_strategy,
        "created_at":         m.created_at,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY-POINTS  (same signatures as original for orchestrate.py)
# ═══════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path:   str,
    output_dir: Optional[str] = None,
    strategy:   str = "hybrid",
    processor:  Optional[DocumentProcessor]  = None,
    chunker:    Optional["IntelligentChunker"] = None,
) -> List[DocumentChunk]:
    """
    Classify → extract → clean → chunk a PDF.
    Saves chunks to JSON if output_dir is given (same contract as original).

    processor / chunker can be passed in from process_document_folder so
    they are reused across all files instead of being recreated each time.
    """
    processor = processor or DocumentProcessor()
    chunker   = chunker   or IntelligentChunker(
        chunk_size=800, chunk_overlap=200, max_chunk_size=1600, strategy=strategy
    )

    document = processor.load_pdf(pdf_path)
    chunks   = chunker.chunk_document(document)

    # Remove near-duplicate chunks caused by double-rendered scanned pages
    # (e.g. docs like EFTA00037195 where native text + OCR produce similar chunks)
    chunks = _dedup_chunks(chunks, similarity_threshold=0.85)

    if output_dir:
        _save_chunks(chunks, pdf_path, output_dir)

    return chunks


def process_text_file(
    txt_path:   str,
    output_dir: Optional[str] = None,
    strategy:   str = "hybrid",
    processor:  Optional[DocumentProcessor]  = None,
    chunker:    Optional["IntelligentChunker"] = None,
) -> List[DocumentChunk]:
    """
    Read → clean → chunk a .txt file.
    Saves chunks to JSON if output_dir is given.

    processor / chunker can be passed in from process_document_folder so
    they are reused across all files instead of being recreated each time.
    """
    processor = processor or DocumentProcessor()
    chunker   = chunker   or IntelligentChunker(
        chunk_size=800, chunk_overlap=200, max_chunk_size=1600, strategy=strategy
    )

    document = processor.load_txt(txt_path)
    if not document.content:
        return []

    chunks = chunker.chunk_document(document)

    # Dedup in case the same content appears multiple times in the text file
    chunks = _dedup_chunks(chunks, similarity_threshold=0.85)

    if output_dir:
        _save_chunks(chunks, txt_path, output_dir)

    return chunks


def process_document_folder(
    folder_path: str,
    output_dir:  Optional[str] = None,
    strategy:    str = "hybrid",
) -> List[DocumentChunk]:
    """
    Process all PDFs and .txt files in a folder (same interface as original).
    Used by orchestrate.py.

    DocumentProcessor and IntelligentChunker are instantiated ONCE here and
    shared across all files — avoids 10 000+ redundant object creations and
    model/tokenizer re-initialisations for large document sets.
    """
    folder = Path(folder_path)
    all_chunks: List[DocumentChunk] = []

    pdf_files = list(folder.glob("**/*.pdf"))
    txt_files = list(folder.glob("**/*.txt"))

    logger.info(f"Found {len(pdf_files)} PDF files, {len(txt_files)} TXT files in {folder}")

    # Instantiate once, reuse across all files
    shared_processor = DocumentProcessor()
    shared_chunker   = IntelligentChunker(
        chunk_size=800, chunk_overlap=200, max_chunk_size=1600, strategy=strategy
    )

    for pdf_file in pdf_files:
        chunks = process_pdf(
            str(pdf_file), output_dir, strategy,
            processor=shared_processor, chunker=shared_chunker,
        )
        all_chunks.extend(chunks)

    for txt_file in txt_files:
        chunks = process_text_file(
            str(txt_file), output_dir, strategy,
            processor=shared_processor, chunker=shared_chunker,
        )
        all_chunks.extend(chunks)

    return all_chunks


# ── Private serialisation helper ───────────────────────────────────────────

def _save_chunks(
    chunks:    List[DocumentChunk],
    src_path:  str,
    output_dir: str,
) -> None:
    """Save chunk list to JSON in the expected format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem        = Path(src_path).stem
    output_file = out / f"{stem}.json"
    data        = [_chunk_to_dict(c) for c in chunks]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"   Saved {len(chunks)} chunks → {output_file}")


# ── CLI convenience ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path       = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed/chunks"
        strategy   = sys.argv[3] if len(sys.argv) > 3 else "hybrid"

        if path.endswith(".txt"):
            chunks = process_text_file(path, output_dir, strategy)
        else:
            chunks = process_pdf(path, output_dir, strategy)

        print(f"\nProcessed {len(chunks)} chunks")
    else:
        print("Usage: python text_pipeline.py <pdf_or_txt_path> [output_dir] [strategy]")