"""
CSV Pipeline: Convert CSV rows to natural language, embed, store
================================================================
Upgraded to incorporate the Advanced RAG notebook's patterns:
  - CSVRecord now mirrors notebook's DocumentMetadata quality fields
  - process_csv logs with notebook emoji style for consistency
  - _assess_record_quality helper (notebook Cell 9 _assess_text_quality style)
  - ingest_csv_to_chromadb uses notebook's EnterpriseVectorStore batch pattern

Original project features kept:
  - All schema-specific converters (flight_log, court_docket, contact_book, investigative)
  - infer_schema from filename / column inspection
  - process_csv / process_csv_folder entry-points (same signatures for orchestrate.py)
  - JSON output format compatible with embeddings.py chunk schema
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS  (extended with notebook-style quality fields)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CSVRecord:
    """
    A single CSV row converted to natural language.

    Changes vs original:
      • quality_score field added (notebook Cell 9 pattern)
      • char_count pre-computed to avoid repeat len() calls
    """
    record_id:     str
    text:          str
    source_file:   str
    row_index:     int
    record_type:   str
    original_data: Dict[str, Any]
    quality_score: float = 0.0    # notebook-style quality tracking
    char_count:    int   = 0
    created_at:    str   = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.char_count:
            self.char_count = len(self.text)


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA-SPECIFIC NL GENERATORS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def flight_log_to_text(row: Dict) -> str:
    """Convert flight log row to natural language sentence."""
    parts = []

    if row.get("date"):
        parts.append(f"On {row['date']}")
    if row.get("passenger"):
        parts.append(f"passenger {row['passenger']} appeared on flight")
    if row.get("aircraft"):
        parts.append(row["aircraft"])
    if row.get("origin") and row.get("destination"):
        parts.append(f"traveling from {row['origin']} to {row['destination']}")

    text = " ".join(parts) + "."

    if row.get("pilot"):
        text += f" The aircraft was piloted by {row['pilot']}."
    if row.get("notes"):
        text += f" Notes: {str(row['notes'])[:200]}"

    return text


def court_docket_to_text(row: Dict) -> str:
    """Convert court docket entry to natural language."""
    text = ""

    if row.get("docketentry_date_filed"):
        text += f"On {row['docketentry_date_filed']}"
    if row.get("docketentry_entry_number"):
        text += f", docket entry #{row['docketentry_entry_number']} was filed."
    else:
        text += " a docket entry was filed."

    if row.get("docketentry_description"):
        desc = str(row["docketentry_description"])[:300]
        text += f" Description: {desc}"
    if row.get("recapdocument_is_sealed"):
        text += " [NOTE: This document is sealed.]"

    return text


def contact_entry_to_text(row: Dict) -> str:
    """Convert contact book entry to natural language."""
    name = row.get("name", "[REDACTED]")
    text = f"{name} appears in the contact book."

    if row.get("address"):
        text += f" Listed address: {row['address']}."
    if row.get("phone"):
        text += f" Phone: {row['phone']}."
    if row.get("email"):
        text += f" Email: {row['email']}."
    if row.get("notes"):
        text += f" Additional notes: {str(row['notes'])[:150]}"

    return text


def investigative_record_to_text(row: Dict) -> str:
    """Generic investigative record converter (fallback)."""
    text = ""

    for key, value in row.items():
        if value and pd.notna(value):
            # Skip ID-like and timestamp columns
            if any(skip in str(key).lower() for skip in ["id", "index", "_id", "timestamp"]):
                continue
            key_display = str(key).replace("_", " ").title()
            text += f"{key_display}: {str(value)[:100]}. "

    return text.strip()


# Schema name → converter mapping (unchanged from original)
SCHEMA_MAP: Dict[str, Callable] = {
    "flight_log":      flight_log_to_text,
    "flight_logs":     flight_log_to_text,
    "court_docket":    court_docket_to_text,
    "docket":          court_docket_to_text,
    "contact_book":    contact_entry_to_text,
    "contacts":        contact_entry_to_text,
    "investigative":   investigative_record_to_text,
    "default":         investigative_record_to_text,
}


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY HELPER  (notebook Cell 9 _assess_text_quality pattern)
# ═══════════════════════════════════════════════════════════════════════════

def _assess_record_quality(text: str, min_chars: int = 20) -> float:
    """
    Quality score 0–1 for a converted CSV record (notebook Cell 9 style).
    Mirrors DocumentProcessor._assess_text_quality applied to short NL sentences.
    """
    if not text:
        return 0.0

    char_count  = len(text)
    word_count  = len(text.split())

    # Length adequacy
    length_score = min(char_count / max(min_chars, 1), 1.0)

    # Reasonable average word length (3–8 → good NL sentence)
    avg_word_len    = char_count / max(word_count, 1)
    word_ratio_score = 1.0 if 3 <= avg_word_len <= 8 else 0.5

    # Low special-character ratio (high ratio = garbled data)
    special = sum(1 for c in text if not c.isalnum() and c not in " .,!?;:-\"\n")
    special_score = 1.0 - min((special / max(len(text), 1)) * 2, 0.5)

    return (length_score + word_ratio_score + special_score) / 3


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA INFERENCE  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def infer_schema(csv_path: str, sample_rows: int = 5) -> str:
    """
    Infer CSV schema from filename and column names.
    Returns a key in SCHEMA_MAP.
    """
    filename = Path(csv_path).stem.lower()

    for schema_name in SCHEMA_MAP:
        if schema_name in filename:
            return schema_name

    try:
        df = pd.read_csv(csv_path, nrows=sample_rows)
        cols = " ".join(df.columns).lower()

        if "flight" in cols or "aircraft" in cols:
            return "flight_log"
        if "docket" in cols or "court" in cols:
            return "court_docket"
        if "contact" in cols or "phone" in cols:
            return "contact_book"
    except Exception:
        pass

    return "investigative"


# ═══════════════════════════════════════════════════════════════════════════
# CORE PROCESSING  (upgraded with notebook patterns)
# ═══════════════════════════════════════════════════════════════════════════

def process_csv(
    csv_path:    str,
    schema_type: Optional[str] = None,
    output_dir:  Optional[str] = None,
) -> List[CSVRecord]:
    """
    Convert a CSV file to naturalised text records.

    Changes vs original:
      • quality_score computed per record (notebook Cell 9 pattern)
      • Summary stats logged in notebook emoji style
      • Output JSON includes quality_score and char_count fields
    """
    csv_path = Path(csv_path)
    logger.info(f"🔄 Processing CSV: {csv_path.name}")

    # Infer schema if not provided
    if schema_type is None:
        schema_type = infer_schema(str(csv_path))
        logger.info(f"   Inferred schema: {schema_type}")

    converter = SCHEMA_MAP.get(schema_type.lower(), SCHEMA_MAP["default"])

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"❌ Error reading CSV {csv_path}: {e}")
        return []

    logger.info(f"   Found {len(df)} rows")

    records: List[CSVRecord] = []

    for idx, row in df.iterrows():
        # Drop NaN values from row dict
        row_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}

        # Convert row to NL text
        try:
            text = converter(row_dict)
        except Exception as e:
            logger.warning(f"⚠️  Row {idx} conversion failed: {e}")
            text = investigative_record_to_text(row_dict)

        if not text or len(text.strip()) < 5:
            continue

        record = CSVRecord(
            record_id     = f"{csv_path.stem}_{idx:06d}",
            text          = text,
            source_file   = str(csv_path),
            row_index     = idx,
            record_type   = schema_type,
            original_data = row_dict,
            # notebook-style quality tracking
            quality_score = _assess_record_quality(text),
        )
        records.append(record)

    # Summary (notebook emoji style)
    avg_quality = sum(r.quality_score for r in records) / max(len(records), 1)
    logger.info(f"✅ {len(records)} records created | avg quality={avg_quality:.2f}")

    # Save to JSON if requested
    if output_dir and records:
        _save_records(records, csv_path, output_dir)

    return records


def process_csv_folder(
    folder_path: str,
    output_dir:  Optional[str] = None,
) -> List[CSVRecord]:
    """Process all CSV files in a folder (same interface as original, used by orchestrate.py)."""
    folder   = Path(folder_path)
    all_recs: List[CSVRecord] = []

    csv_files = list(folder.glob("**/*.csv"))
    logger.info(f"📂 Found {len(csv_files)} CSV files in {folder}")

    for csv_file in csv_files:
        recs = process_csv(str(csv_file), output_dir=output_dir)
        all_recs.extend(recs)

    return all_recs


# ═══════════════════════════════════════════════════════════════════════════
# CHROMADB INGESTION  (upgraded to notebook batch pattern from Cell 14)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_csv_to_chromadb(
    records:         List[CSVRecord],
    collection_name: str,
    embedding_model=None,
    batch_size:      int = 100,
) -> Optional[Any]:
    """
    Ingest CSV records into ChromaDB using notebook's batch-with-stats pattern (Cell 14).

    Changes vs original:
      • batch_size parameter (notebook Cell 14 add_chunks loop)
      • Per-batch stats tracking (successful_adds / failed_adds / batches_processed)
      • Detailed stats logging in notebook emoji style
    """
    import chromadb
    from chromadb.config import Settings

    if embedding_model is None:
        logger.error("❌ Embedding model required for ChromaDB ingestion")
        return None

    try:
        client = chromadb.PersistentClient(
            path="./vectorstore/chroma_db",
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"type": "csv_records"},
        )

        stats = {
            "total_chunks":       len(records),
            "successful_adds":    0,
            "failed_adds":        0,
            "batches_processed":  0,
        }

        texts       = [r.text for r in records]
        total_batches = (len(records) + batch_size - 1) // batch_size

        logger.info(f"🔄 Ingesting {len(records)} records | batch_size={batch_size}")

        for i in range(0, len(records), batch_size):
            batch     = records[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"   Processing batch {batch_num}/{total_batches}")

            try:
                batch_texts = [r.text for r in batch]

                # Embed with 'passage: ' prefix (e5 model convention)
                embeddings = embedding_model.encode(
                    [f"passage: {t}" for t in batch_texts],
                    normalize_embeddings=True,
                ).tolist()

                metadatas = [
                    {
                        "record_type":   r.record_type,
                        "source_file":   r.source_file,
                        "row_index":     r.row_index,
                        # notebook-style: quality info in metadata
                        "quality_score": str(round(r.quality_score, 4)),
                        "char_count":    str(r.char_count),
                    }
                    for r in batch
                ]

                collection.add(
                    documents  = batch_texts,
                    embeddings = embeddings,
                    metadatas  = metadatas,
                    ids        = [r.record_id for r in batch],
                )

                stats["successful_adds"]   += len(batch)
                stats["batches_processed"] += 1

            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                stats["failed_adds"] += len(batch)

        logger.info(
            f"✅ ChromaDB ingestion done | "
            f"successful={stats['successful_adds']}, "
            f"failed={stats['failed_adds']}, "
            f"batches={stats['batches_processed']}"
        )
        return collection

    except Exception as e:
        logger.error(f"❌ ChromaDB ingestion error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _save_records(
    records:   List[CSVRecord],
    src_path:  Path,
    output_dir: str,
) -> None:
    """
    Persist records to JSON in a format compatible with embeddings.py chunk schema.
    The 'text' and 'chunk_id' / 'record_id' fields are preserved for indexing.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_file = out / f"{src_path.stem}.json"
    data = [
        {
            **asdict(r),
            "original_data": r.original_data,
        }
        for r in records
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"   💾 Saved → {output_file}")


# ── CLI convenience ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path   = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed/chunks"
        records    = process_csv(csv_path, output_dir=output_dir)
        print(f"\nProcessed {len(records)} records from CSV")
    else:
        print("Usage: python csv_pipeline.py <csv_path> [output_dir]")