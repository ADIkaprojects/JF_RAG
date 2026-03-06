"""
Image Pipeline — Local Inference Version
Uses local models:
  - openai/clip-vit-base-patch32  → image embeddings only (captioning REMOVED)
  - geolocal/StreetCLIP           → geolocation (zero-shot image classification)
  - llama3.2-vision:latest        → natural-language caption generation (via Ollama)
  - face_recognition              → face detection + identity matching

CHANGES vs previous version:
  - CLIP zero-shot pseudo-captioning REMOVED entirely (fixed 20-label list gone)
  - get_clip_caption() REPLACED by get_llama_caption() using Ollama local inference
  - Face identification added: detect faces, compare against known_faces directory,
    store identified_people list in each image chunk
  - Image embedding logic (CLIP 512-dim) unchanged
  - StreetCLIP geolocation unchanged
  - Chunk schema extended with "identified_people" field
  - All persistence artifacts updated (full metadata, lite metadata, FAISS chunks)

No HF API calls after first download. All inference runs locally.
"""

import json
import logging
import base64
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
import io

import fitz  # pymupdf
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline

from lib.file_resolver import resolve_file


def to_r2_key(path: str) -> str:
    """Normalise a local path to an R2 object key."""
    return str(path).replace("\\", "/").lstrip("./")

logger = logging.getLogger(__name__)

# ── Model Config ─────────────────────────────────────────────────────────────
CLIP_MODEL_NAME       = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
STREETCLIP_MODEL_NAME = "geolocal/StreetCLIP"

# Known faces directory — configurable via env var, defaults to project-relative path
KNOWN_FACES_DIR = os.getenv(
    "KNOWN_FACES_DIR",
    str(Path(__file__).parent.parent / "known_faces")
)

# Face identity confidence thresholds
FACE_HIGH_CONFIDENCE   = 0.80   # similarity >= 0.80 → "high"
FACE_MEDIUM_CONFIDENCE = 0.65   # 0.65–0.79 → "medium"; < 0.65 → discard

# Ollama endpoint for LLaMA 3.2 Vision
OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLAMA_VISION_MODEL = os.getenv("LLAMA_VISION_MODEL", "moondream")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Candidate geo-labels for StreetCLIP (unchanged) ──────────────────────────
GEO_CANDIDATES = [
    "Tokyo, Japan", "New York, USA", "Paris, France", "London, UK",
    "Mumbai, India", "Sydney, Australia", "Berlin, Germany", "Dubai, UAE",
    "São Paulo, Brazil", "Cairo, Egypt", "Moscow, Russia", "Beijing, China",
    "Los Angeles, USA", "Toronto, Canada", "Singapore, Singapore",
    "Bangkok, Thailand", "Mexico City, Mexico", "Istanbul, Turkey",
    "Lagos, Nigeria", "Johannesburg, South Africa", "Amsterdam, Netherlands",
    "Madrid, Spain", "Rome, Italy", "Seoul, South Korea", "Jakarta, Indonesia",
]

# ── Lazy Model Registry ───────────────────────────────────────────────────────
_models: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# CLIP — IMAGE EMBEDDING ONLY  (captioning portion removed entirely)
# ═════════════════════════════════════════════════════════════════════════════

def _get_clip() -> tuple:
    """Load openai/clip-vit-base-patch32 model + processor once, reuse thereafter."""
    if "clip_model" not in _models:
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME} ...")
        _models["clip_processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _models["clip_model"]     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _models["clip_model"].eval()
        logger.info(f"CLIP model loaded on {DEVICE}.")
    return _models["clip_model"], _models["clip_processor"]


def _get_streetclip_pipeline():
    """Load geolocal/StreetCLIP zero-shot-classification pipeline once, reuse thereafter."""
    if "streetclip" not in _models:
        logger.info(f"Loading StreetCLIP model: {STREETCLIP_MODEL_NAME} ...")
        device_id = 0 if torch.cuda.is_available() else -1
        _models["streetclip"] = pipeline(
            "zero-shot-image-classification",
            model=STREETCLIP_MODEL_NAME,
            device=device_id,
        )
        logger.info("StreetCLIP model loaded.")
    return _models["streetclip"]


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw image bytes to a PIL Image in RGB mode."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def get_clip_embedding(pil_image: Image.Image) -> list | None:
    """
    Get CLIP image embedding using openai/clip-vit-base-patch32 locally.
    Returns a normalised list of floats (512-dim for ViT-B/32).

    NOTE: Only the embedding is computed here. CLIP captioning has been
    removed entirely and replaced by get_llama_caption().
    """
    try:
        model, processor = _get_clip()

        inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # vision_model returns BaseModelOutputWithPooling;
            # pooler_output is the projected [CLS] token — correct embedding to use
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_outputs.pooler_output)

        # L2-normalise so embeddings are comparable via dot-product / cosine
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.squeeze(0).cpu().tolist()

    except Exception as e:
        logger.error(f"CLIP embedding failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — LLAMA 3.2 VISION CAPTIONING
# Replaces CLIP pseudo-captioning entirely.
# CAPTION_CANDIDATES list and get_clip_caption() are gone.
# ═════════════════════════════════════════════════════════════════════════════

# System and user prompts enforce the captioning rules from the spec:
# 1–2 sentences, visible content only, no identity/location/intent guessing.
_CAPTION_SYSTEM_PROMPT = (
    "You are a precise image description assistant. "
    "Describe only what is visibly present in the image. "
    "Do not guess identities, locations, or intent. "
    "Do not make emotional interpretations. "
    "Respond with 1-2 sentences maximum."
)

_CAPTION_USER_PROMPT = (
    "Describe this image concisely in 1-2 sentences. "
    "State what is visible: the type of content (document, photograph, "
    "scanned page, etc.) and any clearly observable details. "
    "Do not guess who any person is, where it was taken, or what it means."
)


def get_llama_caption(pil_image: Image.Image) -> str:
    """
    Generate a natural-language caption using llama3.2-vision:latest via Ollama.

    This replaces the old CLIP zero-shot pseudo-captioning entirely.
    The caption is generated (not selected from a fixed list), is grounded
    in visible content only, and is limited to 1–2 sentences per the spec.

    The image is JPEG-encoded and base64-transmitted to the Ollama /api/chat
    endpoint with the multimodal message format.

    Falls back to "Caption unavailable" if:
      - Ollama is not running / unreachable
      - The model is not pulled
      - Any generation error occurs

    Args:
        pil_image: PIL Image in RGB mode

    Returns:
        A 1–2 sentence natural-language description string.
    """
    try:
        # Encode PIL image to base64 JPEG for Ollama multimodal API
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "model": LLAMA_VISION_MODEL,
            "messages": [
                {
                    "role":    "system",
                    "content": _CAPTION_SYSTEM_PROMPT,
                },
                {
                    "role":    "user",
                    "content": _CAPTION_USER_PROMPT,
                    "images":  [image_b64],
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,    # low temperature for deterministic, factual output
                "num_predict": 60,     # reduced from 120 — 1–2 sentences, faster on CPU
            },
        }

        data = json.dumps(payload).encode("utf-8")
        url  = f"{OLLAMA_HOST}/api/chat"

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=300) as resp:  # 5 min — CPU inference is slow
            response_body = json.loads(resp.read().decode("utf-8"))

        caption = response_body.get("message", {}).get("content", "").strip()

        if not caption:
            logger.warning("LLaMA Vision returned empty caption — using fallback")
            return "Caption unavailable"

        # Enforce 1–2 sentence hard limit: split on sentence boundaries and keep first two
        sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
        caption   = " ".join(sentences[:2]).strip()

        logger.debug(f"LLaMA caption: '{caption}'")
        return caption

    except urllib.error.URLError as e:
        logger.error(
            f"Ollama unreachable at {OLLAMA_HOST}: {e} — "
            f"ensure Ollama is running and '{LLAMA_VISION_MODEL}' is pulled"
        )
        return "Caption unavailable"
    except Exception as e:
        logger.error(f"LLaMA Vision captioning failed: {e}")
        return "Caption unavailable"


# ═════════════════════════════════════════════════════════════════════════════
# STREETCLIP GEOLOCATION  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def get_streetclip_geolocation(pil_image: Image.Image) -> dict:
    """
    Get geolocation estimate using geolocal/StreetCLIP locally.
    Runs zero-shot-image-classification against GEO_CANDIDATES.
    Returns a dict with city, country, confidence, and top-3 predictions.
    UNCHANGED from original.
    """
    try:
        clf     = _get_streetclip_pipeline()
        results = clf(pil_image, candidate_labels=GEO_CANDIDATES)

        if not results:
            return {"city": None, "country": None, "confidence": None}

        top   = results[0]
        label = top.get("label", "")
        score = round(top.get("score", 0.0), 3)

        parts   = label.split(",")
        city    = parts[0].strip() if len(parts) > 0 else None
        country = parts[1].strip() if len(parts) > 1 else None

        return {
            "city"           : city,
            "country"        : country,
            "confidence"     : score,
            "raw_label"      : label,
            "all_predictions": results[:3],
        }

    except Exception as e:
        logger.error(f"StreetCLIP geolocation failed: {e}")
        return {"city": None, "country": None, "confidence": None}


# ═════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — FACE IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def _get_face_recognition_lib():
    """
    Lazy-load the face_recognition library.
    Returns the module on success or None if not installed.
    """
    if "face_recognition" not in _models:
        try:
            import face_recognition as fr
            _models["face_recognition"] = fr
            logger.info("face_recognition library loaded.")
        except ImportError:
            logger.error(
                "face_recognition library not installed. "
                "Install with: pip install face-recognition  "
                "(requires dlib; see https://github.com/ageitgey/face_recognition)"
            )
            _models["face_recognition"] = None
    return _models["face_recognition"]


def _name_from_filename(filepath: str) -> str:
    """
    Derive a human-readable display name from a known-faces filename.

    Convention: underscores are word separators, each word is title-cased.
      'alan_dershowitz.png'  → 'Alan Dershowitz'
      'ghislaine_maxwell.jpg' → 'Ghislaine Maxwell'
    """
    stem  = Path(filepath).stem                   # 'alan_dershowitz'
    parts = stem.replace("-", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p)


def _confidence_label(similarity: float) -> str | None:
    """
    Map a cosine similarity score to a confidence label per the spec.

    >= 0.80  → 'high'
    0.65–0.79 → 'medium'
    < 0.65   → None  (caller must discard — do not record)
    """
    if similarity >= FACE_HIGH_CONFIDENCE:
        return "high"
    if similarity >= FACE_MEDIUM_CONFIDENCE:
        return "medium"
    return None   # below threshold — discard entirely


def load_known_face_encodings(known_faces_dir: str) -> list[dict]:
    """
    Load and encode all reference faces from the known_faces directory.

    Each image file in the directory must contain exactly one clearly visible
    face. Files that produce no detectable face are skipped with a warning.
    Files with multiple faces use only the first face and log a warning.

    This function is called ONCE before the processing loop in
    process_image_folder() — it is expensive and must not be repeated per image.

    Args:
        known_faces_dir: path to directory containing reference face images
                         named as full_name.png / full_name.jpg

    Returns:
        List of dicts:
        [{"name": "Alan Dershowitz", "encoding": np.ndarray (128-dim)}, ...]
    """
    fr = _get_face_recognition_lib()
    if fr is None:
        return []

    known_dir = Path(known_faces_dir)
    if not known_dir.exists():
        logger.warning(f"Known faces directory not found: {known_faces_dir}")
        return []

    supported   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in sorted(known_dir.iterdir()) if f.suffix.lower() in supported]

    if not image_files:
        logger.warning(f"No image files found in known_faces directory: {known_faces_dir}")
        return []

    known_people: list[dict] = []

    for img_file in image_files:
        name = _name_from_filename(str(img_file))
        try:
            known_image = fr.load_image_file(str(img_file))
            encodings   = fr.face_encodings(known_image)

            if not encodings:
                logger.warning(
                    f"No face detected in known_faces/{img_file.name} "
                    f"— skipping '{name}'"
                )
                continue

            if len(encodings) > 1:
                logger.warning(
                    f"Multiple faces in known_faces/{img_file.name} "
                    f"— using first face for '{name}'"
                )

            known_people.append({
                "name":     name,
                "encoding": encodings[0],   # 128-dim numpy array (dlib HOG descriptor)
            })
            logger.debug(f"   Known face loaded: '{name}'")

        except Exception as e:
            logger.warning(f"Failed to load known face '{img_file.name}': {e}")

    logger.info(
        f"Loaded {len(known_people)} known face(s) from {known_faces_dir}"
    )
    return known_people


def identify_faces(
    pil_image:    Image.Image,
    known_people: list[dict],
) -> list[dict]:
    """
    Detect all faces in pil_image and attempt to match each against known_people.

    Algorithm:
      1. Convert PIL image to RGB numpy array (required by face_recognition)
      2. Detect all face bounding boxes using HOG model
      3. Compute 128-dim dlib encodings for every detected face
      4. For each detected face:
           a. Compute Euclidean distance to every known-face encoding
           b. Convert distance to similarity: sim = clip(1 - distance, 0, 1)
              Mapping rationale:
                distance 0.00 → similarity 1.00 (identical)
                distance 0.20 → similarity 0.80 (high confidence boundary)
                distance 0.35 → similarity 0.65 (medium confidence boundary)
                distance 0.60 → similarity 0.40 (standard library reject threshold)
           c. Best match = lowest Euclidean distance = highest similarity
           d. Apply confidence threshold: discard if similarity < 0.65
      5. Return list of confirmed identities (one entry per matched face)

    Supports:
      - Zero faces  → returns []
      - One face    → returns list with at most one entry
      - Many faces  → returns list with one entry per successfully matched face

    Face embeddings are NOT stored in the returned dicts — only name, similarity,
    and confidence label are recorded per the spec.

    Args:
        pil_image:    RGB PIL image to analyse
        known_people: output of load_known_face_encodings()

    Returns:
        [{"name": "...", "similarity": float, "confidence": "high"|"medium"}, ...]
    """
    fr = _get_face_recognition_lib()
    if fr is None or not known_people:
        return []

    import numpy as np

    try:
        img_array = np.array(pil_image.convert("RGB"))

        # HOG face detection — faster than CNN, no GPU needed, sufficient accuracy
        face_locations = fr.face_locations(img_array, model="hog")

        if not face_locations:
            logger.debug("No faces detected in image.")
            return []

        logger.debug(f"Detected {len(face_locations)} face(s) in image.")

        # Compute 128-dim encodings for all detected faces in one call
        face_encodings = fr.face_encodings(
            img_array,
            known_face_locations=face_locations,
        )

        known_encodings = [p["encoding"] for p in known_people]
        identified: list[dict] = []

        for face_enc in face_encodings:
            # face_distance returns Euclidean distances; lower = more similar
            distances    = fr.face_distance(known_encodings, face_enc)
            similarities = np.clip(1.0 - distances, 0.0, 1.0)

            best_idx    = int(np.argmin(distances))
            best_sim    = float(similarities[best_idx])
            confidence  = _confidence_label(best_sim)

            if confidence is None:
                # Below minimum threshold — do not record this face
                logger.debug(
                    f"Best candidate '{known_people[best_idx]['name']}' "
                    f"similarity={best_sim:.3f} — below threshold, discarded"
                )
                continue

            identified.append({
                "name":       known_people[best_idx]["name"],
                "similarity": round(best_sim, 4),
                "confidence": confidence,
            })
            logger.debug(
                f"   Matched: '{known_people[best_idx]['name']}' "
                f"similarity={best_sim:.3f} ({confidence})"
            )

        return identified

    except Exception as e:
        logger.error(f"Face identification failed: {e}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION FROM PDF  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def extract_images_from_pdf(
    pdf_path:   str,
    output_dir: str,
    render_dpi: int = 150,
) -> list[str]:
    """
    Extract images from a PDF using pymupdf.

    Strategy:
      1. Try to extract embedded image XObjects from each page (fast, lossless).
      2. If a page has NO embedded images (e.g. scanned PDFs where the whole
         page IS the image), render that page at `render_dpi` and save as PNG.

    Returns list of saved image file paths.
    In production, the PDF bytes may be loaded from Cloudflare R2 via file_resolver.
    """
    # Use file_resolver so PDFs can be stored in R2 in production.
    # Normalise the path so keys use forward slashes on Windows.
    pdf_bytes = resolve_file(to_r2_key(pdf_path))
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    pdf_name    = Path(pdf_path).stem

    for page_num in range(len(doc)):
        page       = doc[page_num]
        image_list = page.get_images(full=True)

        if image_list:
            # ── Mode 1: extract embedded image XObjects ───────────────────
            for img_index, xref_data in enumerate(image_list):
                xref = xref_data[0]
                try:
                    img_data  = doc.extract_image(xref)
                    img_bytes = img_data["image"]
                    ext       = img_data["ext"]

                    filename = f"{pdf_name}_p{page_num + 1}_img{img_index + 1}.{ext}"
                    fpath    = out / filename

                    with open(fpath, "wb") as f:
                        f.write(img_bytes)

                    saved_paths.append(str(fpath))
                    logger.debug(f"Extracted embedded image: {filename}")

                except Exception as e:
                    logger.warning(
                        f"Failed to extract image xref={xref} from {pdf_path}: {e}"
                    )
        else:
            # ── Mode 2: render full page (scanned PDF / no embedded images)
            try:
                mat      = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                pix      = page.get_pixmap(matrix=mat, alpha=False)
                filename = f"{pdf_name}_p{page_num + 1}_rendered.png"
                fpath    = out / filename
                pix.save(str(fpath))
                saved_paths.append(str(fpath))
                logger.debug(f"Rendered page as image: {filename}")
            except Exception as e:
                logger.warning(
                    f"Failed to render page {page_num + 1} of {pdf_path}: {e}"
                )

    doc.close()
    logger.info(
        f"Extracted {len(saved_paths)} images from {Path(pdf_path).name}"
    )
    return saved_paths


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE PROCESSOR  (updated — new caption + face ID added)
# ═════════════════════════════════════════════════════════════════════════════

def process_single_image(
    image_path:   str,
    source_doc:   str,
    known_people: list[dict],
) -> dict | None:
    """
    Process one image locally:
      1. Read bytes → decode to PIL once (shared across all inference calls)
      2. CLIP embedding          (UNCHANGED — 512-dim, L2-normalised)
      3. LLaMA 3.2 Vision caption (NEW — replaces CLIP pseudo-captioning)
      4. StreetCLIP geolocation  (UNCHANGED)
      5. Face identification      (NEW — compares against known_people)
      6. Return metadata dict, or None on critical failure

    The returned dict matches the updated chunk schema exactly:
      {
          "image_path":        str,
          "source_doc":        str,
          "caption":           str,       # generated, not selected
          "embedding":         list[float],
          "geo_estimate":      dict,
          "identified_people": list[dict],  # NEW FIELD
          "source_type":       "IMAGE"
      }

    Args:
        image_path:   absolute path to the extracted image file
        source_doc:   PDF stem (e.g. "report" from "report_p1_img1.jpg")
        known_people: pre-loaded list from load_known_face_encodings()
    """
    logger.info(f"  Processing image: {Path(image_path).name}")

    # ── Read raw bytes ───────────────────────────────────────────────────
    try:
        # Use file_resolver so images can be read from R2 in production.
        image_bytes = resolve_file(image_path)
    except Exception as e:
        logger.error(f"Could not read image {image_path}: {e}")
        return None

    # Decode once, reuse across all three inference calls
    try:
        pil_image = _bytes_to_pil(image_bytes)
    except Exception as e:
        logger.error(f"Could not decode image {image_path}: {e}")
        return None

    # ── CLIP embedding (unchanged) ───────────────────────────────────────
    embedding = get_clip_embedding(pil_image)
    if embedding is None:
        logger.warning(f"Skipping {image_path} — CLIP embedding failed")
        return None

    # ── LLaMA 3.2 Vision caption (replaces CLIP pseudo-captioning) ───────
    caption = get_llama_caption(pil_image)

    # ── StreetCLIP geolocation (unchanged) ──────────────────────────────
    geo = get_streetclip_geolocation(pil_image)

    # ── Face identification (new) ────────────────────────────────────────
    identified_people = identify_faces(pil_image, known_people)

    return {
        "image_path"       : image_path,
        "source_doc"       : source_doc,
        "caption"          : caption,
        "embedding"        : embedding,
        "geo_estimate"     : geo,
        "identified_people": identified_people,   # NEW FIELD per spec
        "source_type"      : "IMAGE",
    }


# ═════════════════════════════════════════════════════════════════════════════
# FOLDER PROCESSOR  (called by orchestrate.py)
# ═════════════════════════════════════════════════════════════════════════════

def process_image_folder(
    image_dir:       str,
    extract_dir:     str,
    metadata_dir:    str,
    known_faces_dir: str = KNOWN_FACES_DIR,
) -> list[dict]:
    """
    Main entry point called by orchestrate.py.

    Steps:
      1. Find all PDFs in image_dir  (only PDFs are processed — unchanged)
      2. Extract embedded images from each PDF
      3. Pre-load known-face encodings from known_faces_dir ONCE before the loop
      4. For each extracted image:
           - CLIP embed (unchanged)
           - LLaMA 3.2 Vision caption (new — replaces CLIP pseudo-captions)
           - StreetCLIP geo (unchanged)
           - Face identification (new)
      5. Save:
           processed/image_meta/image_metadata.json      (full: embeddings + face IDs)
           processed/image_meta/image_metadata_lite.json (lite: captions + face IDs, no embeddings)
      6. Return list of all metadata dicts

    The known_faces_dir parameter accepts an override for testing / different
    machines; defaults to the fixed spec path D:\\JF_RAG\\known_faces.

    Signature change vs original:
      Added optional `known_faces_dir` kwarg — orchestrate.py call is unchanged
      since it passes only the three positional args.
    """
    image_dir_path    = Path(image_dir)
    metadata_dir_path = Path(metadata_dir)
    metadata_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Pre-load heavy models once before the loop ────────────────────────
    logger.info("Pre-loading local models...")
    _get_clip()
    _get_streetclip_pipeline()
    logger.info("CLIP + StreetCLIP models ready.")
    logger.info(f"LLaMA Vision captions will be generated via Ollama at: {OLLAMA_HOST}")

    # ── Pre-load known-face encodings once (expensive — dlib HOG per image) ─
    logger.info(f"Loading known faces from: {known_faces_dir}")
    known_people = load_known_face_encodings(known_faces_dir)
    if not known_people:
        logger.warning(
            "No known faces loaded — face identification will be skipped. "
            "All 'identified_people' fields will be empty lists."
        )

    # ── Step 1: Extract images from all PDFs ─────────────────────────────
    pdf_files = list(image_dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {image_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in {image_dir}")

    all_image_paths: list[str] = []
    for pdf_file in pdf_files:
        logger.info(f"Extracting images from: {pdf_file.name}")
        extracted = extract_images_from_pdf(str(pdf_file), extract_dir)
        all_image_paths.extend(extracted)

    logger.info(f"Total images extracted: {len(all_image_paths)}")

    if not all_image_paths:
        logger.warning("No images found inside the PDFs.")
        return []

    # ── Step 2: Process each image ────────────────────────────────────────
    all_metadata: list[dict] = []
    failed:       list[str]  = []

    logger.info(f"Processing {len(all_image_paths)} images...")

    for i, image_path in enumerate(all_image_paths, 1):
        logger.info(f"[{i}/{len(all_image_paths)}] {Path(image_path).name}")

        # source_doc = PDF stem, e.g. "report" from "report_p1_img1.jpg"
        source_doc = Path(image_path).stem.split("_p")[0]

        metadata = process_single_image(
            image_path   = image_path,
            source_doc   = source_doc,
            known_people = known_people,
        )

        if metadata:
            all_metadata.append(metadata)
        else:
            failed.append(image_path)

    # ── Step 3: Persist metadata ──────────────────────────────────────────
    full_meta_path = metadata_dir_path / "image_metadata.json"       # full: embeddings + face IDs
    lite_meta_path = metadata_dir_path / "image_metadata_lite.json"  # lite: captions + face IDs only

    # Lite version: strip embedding arrays but KEEP captions AND identified_people
    lite_meta = [
        {k: v for k, v in m.items() if k != "embedding"}
        for m in all_metadata
    ]

    with open(full_meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    with open(lite_meta_path, "w", encoding="utf-8") as f:
        json.dump(lite_meta, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    total_faces_found = sum(len(m.get("identified_people", [])) for m in all_metadata)

    logger.info("=" * 50)
    logger.info(f"✅ Images processed successfully : {len(all_metadata)}")
    logger.info(f"👤 Total face identifications   : {total_faces_found}")
    if failed:
        logger.warning(f"❌ Images failed               : {len(failed)}")
        for f_path in failed:
            logger.warning(f"   - {Path(f_path).name}")
    logger.info(f"📁 Full metadata saved to      : {full_meta_path}")
    logger.info(f"📁 Lite metadata saved to      : {lite_meta_path}")
    logger.info("=" * 50)

    return all_metadata