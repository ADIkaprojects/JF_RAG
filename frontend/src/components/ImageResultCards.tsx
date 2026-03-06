/**
 * ImageResultCards
 * ─────────────────────────────────────────────────────────────────────────────
 * Renders a horizontal strip of up to 3 image result cards above the assistant
 * answer bubble. Only mounted when:
 *   1. The user query contained a visual-intent keyword (imageQueryDetected)
 *   2. The backend returned at least one image source (imageSources.length > 0)
 *
 * Each card shows:
 *   • Thumbnail — served via FastAPI's /images/{path} static mount
 *   • Caption   — 2-line clamp, full text on hover via tooltip
 *   • Geo label — city name from StreetCLIP (if present)
 *   • Match %   — cosine similarity score from CLIP FAISS search
 *
 * Cards animate in with a staggered entrance (0 / 80 / 160 ms delay).
 * Clicking a card opens a lightbox overlay showing the full-size image.
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MapPin, X, ZoomIn, ImageOff } from "lucide-react";
import { BASE_URL, type RetrievalResult } from "@/lib/api";

interface ImageResultCardsProps {
  /** Up to 3 image retrieval results from QueryResponse.image_sources */
  sources: RetrievalResult[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Build the URL the browser uses to fetch the thumbnail.
 * image_path from the backend can be:
 *   • A Windows absolute path  →  D:\JF_RAG\processed\images\foo.png
 *   • A POSIX relative path    →  processed/images/foo.png
 *   • Just a filename          →  foo.png
 *
 * The FastAPI mount serves everything under /images/ from ./processed/images/,
 * so we only need the bare filename.
 */
function buildImageUrl(imagePath: string): string {
  // Normalise backslashes to forward slashes
  const normalised = imagePath.replace(/\\/g, "/");
  // Take only the last path segment (filename)
  const filename = normalised.split("/").pop() ?? normalised;
  return `${BASE_URL}/images/${encodeURIComponent(filename)}`;
}

function formatSimilarity(sim: number | undefined): string {
  if (sim === undefined || sim === null) return "";
  // CLIP FAISS returns L2 distances (lower = better) OR cosine scores
  // (higher = better).  Values > 1 are L2 distances; clamp and invert for %.
  const pct = sim <= 1 ? sim * 100 : Math.max(0, 100 - sim * 10);
  return `${Math.min(100, Math.round(pct))}% match`;
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface CardProps {
  source: RetrievalResult;
  index: number;
  onExpand: (src: RetrievalResult) => void;
}

const ImageCard = ({ source, index, onExpand }: CardProps) => {
  const [imgError, setImgError] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);
  
  // Safely get image path with fallback
  const imagePath = source?.image_path ?? source?.chunk_id ?? "";
  const url = imagePath ? buildImageUrl(imagePath) : "";

  // Skip rendering if we don't have a valid source
  if (!source || !imagePath) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, delay: index * 0.08, ease: "easeOut" }}
      className="group relative flex-1 min-w-0 rounded-2xl overflow-hidden cursor-pointer"
      style={{
        // Use theme-aware feature card tokens so cards adapt to light/dark mode.
        background: "hsl(var(--feature-card-bg))",
        border: "1px solid hsl(var(--feature-card-border))",
        maxWidth: "200px",
      }}
      onClick={() => onExpand(source)}
      whileHover={{ y: -2, transition: { duration: 0.15 } }}
    >
      {/* Thumbnail */}
      <div className="relative w-full h-[110px] overflow-hidden"
        style={{ background: "hsl(var(--feature-card-border))" }}>
        {!imgError ? (
          <>
            {/* Shimmer placeholder while loading */}
            {!imgLoaded && (
              <div className="absolute inset-0 shimmer" />
            )}
            <img
              src={url}
              alt={source.caption ?? "Image result"}
              className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
              style={{ opacity: imgLoaded ? 1 : 0, transition: "opacity 0.2s" }}
              onLoad={() => setImgLoaded(true)}
              onError={() => setImgError(true)}
            />
          </>
        ) : (
          /* Fallback when image can't be served */
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
            <ImageOff size={20} className="text-muted-foreground/30" />
            <span className="text-[10px] text-muted-foreground/40">No preview</span>
          </div>
        )}

        {/* Zoom-in hint on hover */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          style={{ background: "hsl(var(--background) / 0.55)" }}>
          <div className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{ background: "hsl(198 90% 56% / 0.9)" }}>
            <ZoomIn size={14} style={{ color: "hsl(220 16% 7%)" }} />
          </div>
        </div>

        {/* Similarity badge — top-right corner */}
        {source.similarity !== undefined && (
          <div className="absolute top-2 right-2 text-[10px] font-semibold px-1.5 py-0.5 rounded-md"
            style={{
              background: "hsl(var(--background) / 0.9)",
              color: "hsl(198 90% 60%)",
              border: "1px solid hsl(198 90% 56% / 0.3)",
              backdropFilter: "blur(4px)",
            }}>
            {formatSimilarity(source.similarity)}
          </div>
        )}
      </div>

      {/* Info section */}
      <div className="px-3 py-2.5">
        {/* Caption — 2 lines max */}
        {source.caption && (
          <p className="text-[11px] text-foreground/80 leading-snug line-clamp-2 mb-1.5">
            {source.caption}
          </p>
        )}

        {/* Geo label */}
        {source.geo_city && (
          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <MapPin size={9} className="shrink-0 text-accent/50" />
            <span className="truncate">{source.geo_city}</span>
          </div>
        )}

        {/* Source filename — very subtle */}
        <p className="text-[10px] text-muted-foreground/35 truncate mt-1">
          {source.source_file?.split(/[\\/]/).pop() ?? source.chunk_id}
        </p>
      </div>
    </motion.div>
  );
};

// ── Lightbox overlay ──────────────────────────────────────────────────────────

interface LightboxProps {
  source: RetrievalResult;
  onClose: () => void;
}

const Lightbox = ({ source, onClose }: LightboxProps) => {
  // Close via effect — never call state-setters during another component's render
  useEffect(() => {
    if (!source || !source.image_path) onClose();
  }, [source, onClose]);

  if (!source || !source.image_path) return null;
  
  const url = buildImageUrl(source.image_path ?? "");

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: "hsl(var(--background) / 0.92)", backdropFilter: "blur(16px)" }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.92, opacity: 0 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
        className="relative max-w-3xl w-full rounded-3xl overflow-hidden"
        style={{
          background: "hsl(var(--feature-card-bg))",
          border: "1px solid hsl(var(--feature-card-border))",
          boxShadow: "0 32px 80px rgba(0,0,0,0.6)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-10 w-8 h-8 rounded-full flex items-center justify-center transition-colors"
          style={{
            background: "hsl(var(--feature-card-border))",
            color: "hsl(var(--muted-foreground))",
          }}
          onMouseEnter={e =>
            (e.currentTarget.style.background = "hsl(var(--feature-card-border) / 1)")
          }
          onMouseLeave={e =>
            (e.currentTarget.style.background = "hsl(var(--feature-card-border))")
          }
        >
          <X size={15} />
        </button>

        {/* Full image */}
        <img
          src={url}
          alt={source.caption ?? "Image result"}
          className="w-full max-h-[65vh] object-contain"
          style={{ background: "hsl(var(--background))" }}
        />

        {/* Metadata footer */}
        <div className="px-5 py-4 space-y-1.5">
          {source.caption && (
            <p className="text-sm text-foreground/85 leading-relaxed">{source.caption}</p>
          )}
          <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
            {source.geo_city && (
              <span className="flex items-center gap-1">
                <MapPin size={11} className="text-accent/50" />
                {source.geo_city}
              </span>
            )}
            {source.similarity !== undefined && (
              <span style={{ color: "hsl(198 90% 60%)" }}>
                {formatSimilarity(source.similarity)}
              </span>
            )}
            <span className="text-muted-foreground/40 font-mono truncate">
              {source.source_file?.split(/[\\/]/).pop() ?? source.chunk_id}
            </span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

// ── Main export ───────────────────────────────────────────────────────────────

const ImageResultCards = ({ sources }: ImageResultCardsProps) => {
  const [lightboxSrc, setLightboxSrc] = useState<RetrievalResult | null>(null);

  // Safety check: ensure sources is an array
  if (!Array.isArray(sources) || sources.length === 0) {
    return null;
  }

  // Show at most 3 cards
  const visible = sources.slice(0, 3).filter(src => src && src.image_path);

  // Final safety check: return null if no valid cards after filtering
  if (visible.length === 0) {
    return null;
  }

  return (
    <>
      {/* Section header */}
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="flex items-center gap-2 mb-2"
      >
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-accent">
          <span className="w-1 h-1 rounded-full bg-accent" />
          {visible.length} image result{visible.length > 1 ? "s" : ""} found
        </div>
        <div className="flex-1 h-px" style={{ background: "hsl(var(--feature-card-border))" }} />
      </motion.div>

      {/* Card strip */}
      <div className="flex gap-3 mb-3 flex-wrap">
        {visible.map((src, i) => (
          <ImageCard
            key={src.chunk_id ?? i}
            source={src}
            index={i}
            onExpand={setLightboxSrc}
          />
        ))}
      </div>

      {/* Lightbox */}
      <AnimatePresence>
        {lightboxSrc && (
          <Lightbox source={lightboxSrc} onClose={() => setLightboxSrc(null)} />
        )}
      </AnimatePresence>
    </>
  );
};

export default ImageResultCards;
