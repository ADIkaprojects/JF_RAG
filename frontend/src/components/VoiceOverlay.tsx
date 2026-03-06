/**
 * VoiceOverlay — Feature 4
 * Full-screen modal wrapper for the VoiceAssistant.
 * Provides backdrop, background arc decorations, Escape key handling,
 * and AnimatePresence-compatible mounting.
 *
 * Does NOT render inside a Route — it mounts directly in Index.tsx.
 */

import { useEffect } from "react";
import { motion } from "framer-motion";
import VoiceAssistant from "./VoiceAssistant";

interface VoiceOverlayProps {
  onClose: () => void;
  onQueryReady: (transcript: string) => Promise<string>;
}

const VoiceOverlay = ({ onClose, onQueryReady }: VoiceOverlayProps) => {

  // Escape key closes the overlay without submitting
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-50 flex items-center justify-center"
    >
      {/* Backdrop — click to close */}
      <div
        className="absolute inset-0"
        style={{ background: "hsl(220 16% 7% / 0.85)", backdropFilter: "blur(8px)" }}
        onClick={onClose}
      />

      {/* Background globe glow */}
      <div className="absolute inset-0 globe-glow pointer-events-none" />

      {/* Arc circle decoration */}
      <div
        className="absolute bottom-0 left-1/2 w-[120%] aspect-square rounded-full pointer-events-none"
        style={{
          transform: "translateX(-50%) translateY(55%)",
          border: "1px solid hsl(217 91% 60% / 0.15)",
          boxShadow: "0 0 40px hsl(217 91% 60% / 0.1), inset 0 0 40px hsl(217 91% 60% / 0.05)",
        }}
      />

      {/* Voice assistant card — stop click propagation so backdrop click doesn't fire */}
      <div className="relative z-10 w-full max-w-md px-4" onClick={(e) => e.stopPropagation()}>
        <VoiceAssistant onClose={onClose} onQueryReady={onQueryReady} />
      </div>
    </motion.div>
  );
};

export default VoiceOverlay;

