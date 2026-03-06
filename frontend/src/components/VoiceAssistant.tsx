import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2 } from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer";
import MicButton from "./MicButton";
import { useVapi } from "@/hooks/useVapi";

interface VoiceAssistantProps {
  onClose?: () => void;
  onQueryReady: (transcript: string) => Promise<string>;
}

/**
 * Cleans RAG response text before it is passed to Vapi for speech.
 * - Removes [Source: ...] citation blocks entirely
 * - Replaces specific document IDs (e.g. "Document EFTA00014822") with "some documents"
 * - Collapses extra whitespace left behind
 */
function sanitiseForSpeech(text: string): string {
  return text
    .replace(/\[Source:[^\]]+\]/g, "")
    .replace(/Documents?\s+EFTA\w+(?:\s*,\s*EFTA\w+)*(?:\s+and\s+EFTA\w+)?/gi, "Some documents")
    .replace(/\bEFTA\w+(?:\.pdf)?\b/g, "")
    .replace(/,\s*Page\s+\d+,\s*Type:\s*\w+/gi, "")
    .replace(/\s{2,}/g, " ")
    .replace(/,\s*,/g, ",")
    .replace(/\.\s*\./g, ".")
    .trim();
}

const VoiceAssistant = ({ onClose, onQueryReady }: VoiceAssistantProps) => {
  const [isListening, setIsListening] = useState(false);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Subtitle state — one line at a time
  const [currentSubtitle, setCurrentSubtitle] = useState<string>("");
  const [subtitleVisible, setSubtitleVisible] = useState(false);
  const subtitleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);

  // Show a subtitle line, then auto-hide after a read-time delay
  const showSubtitle = useCallback((line: string) => {
    if (!line.trim()) return;
    if (subtitleTimerRef.current) clearTimeout(subtitleTimerRef.current);
    setCurrentSubtitle(line);
    setSubtitleVisible(true);
    const words = line.split(" ").length;
    const displayMs = Math.max(1500, words * 400);
    subtitleTimerRef.current = setTimeout(() => setSubtitleVisible(false), displayMs);
  }, []);

  const handleSpeechEnd = useCallback(() => {
    if (subtitleTimerRef.current) clearTimeout(subtitleTimerRef.current);
    subtitleTimerRef.current = setTimeout(() => setSubtitleVisible(false), 1200);
  }, []);

  const { startCall, speakText, stopCall, isSpeaking } = useVapi({
    onSubtitleLine: showSubtitle,
    onSpeechEnd: handleSpeechEnd,
  });

  // Keep a ref so event handlers can read isSpeaking without stale closure
  const isSpeakingRef = useRef(false);
  useEffect(() => {
    isSpeakingRef.current = isSpeaking;
  }, [isSpeaking]);

  // ─── Hard mic-lock ───────────────────────────────────────────────────────────
  // The moment Vapi starts speaking, forcibly abort any active recognition so
  // her voice cannot be captured as a user query.
  useEffect(() => {
    if (isSpeaking && recognitionRef.current) {
      recognitionRef.current.abort();
      recognitionRef.current = null;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      setAnalyserNode(null);
      setIsListening(false);
    }
  }, [isSpeaking]);

  // ─── Start Vapi call on mount → she greets the user automatically ────────────
  useEffect(() => {
    startCall();
    showSubtitle("Connecting to Jeffrey's Archive…");
    return () => {
      stopCall();
      if (subtitleTimerRef.current) clearTimeout(subtitleTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setAnalyserNode(null);
    setIsListening(false);
  }, []);

  const startListening = useCallback(async () => {
    // Tapping mic while Vapi is speaking = interrupt her
    if (isSpeakingRef.current) {
      stopCall();
      showSubtitle("Go ahead, I'm listening…");
      // Small pause so stopCall has time to clear isSpeaking before recognition starts
      await new Promise((res) => setTimeout(res, 150));
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      setAnalyserNode(analyser);

      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (!SpeechRecognition) return;

      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      let latestTranscript = "";

      recognition.onresult = (event: any) => {
        // Safety net: if Vapi somehow started speaking again, abort immediately
        if (isSpeakingRef.current) {
          recognition.abort();
          return;
        }

        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        latestTranscript = transcript;
        showSubtitle(transcript);

        const isFinal = event.results[event.results.length - 1].isFinal;
        if (isFinal) {
          const finalTranscript = latestTranscript.trim();
          stopListening();
          if (!finalTranscript) return;

          setIsSubmitting(true);
          showSubtitle("Searching the archive…");

          onQueryReady(finalTranscript)
            .then(async (answer) => {
              if (!answer?.trim()) return;
              const spokenAnswer = sanitiseForSpeech(answer);
              if (!spokenAnswer.trim()) return;
              await speakText(spokenAnswer);
            })
            .catch((err: any) => {
              showSubtitle(`Error: ${err?.message ?? "Request failed"}`);
            })
            .finally(() => {
              setIsSubmitting(false);
            });
        }
      };

      recognition.onerror = (event: any) => {
        // "aborted" is our intentional mic-lock — not a real error
        if (event.error === "aborted") return;
        console.error("Speech recognition error:", event.error);
        // Always clean up so the UI doesn't get stuck in "listening" state
        stopListening();
      };

      recognition.start();
      recognitionRef.current = recognition;
      setIsListening(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }, [onQueryReady, stopListening, stopCall, showSubtitle, speakText]);

  const toggleListening = useCallback(() => {
    if (isListening) stopListening();
    else startListening();
  }, [isListening, startListening, stopListening]);

  const handleClose = useCallback(() => {
    stopListening();
    stopCall();
    onClose?.();
  }, [stopListening, stopCall, onClose]);

  // Contextual label under the mic button
  const micLabel = isSpeaking
    ? "Tap to interrupt"
    : isListening
    ? "Listening…"
    : isSubmitting
    ? "Searching…"
    : "Tap to speak";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: 20 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="voice-glass-card dot-pattern rounded-3xl w-full max-w-md relative overflow-hidden"
    >
      {/* Close button */}
      <div className="absolute top-4 right-4 z-10">
        {onClose && (
          <button
            onClick={handleClose}
            className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center hover:bg-secondary/80 transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
      </div>

      <div className="p-8 pt-10 pb-6 flex flex-col items-center gap-6">

        {/* ── Subtitle display — one line at a time ── */}
        <div className="w-full min-h-[80px] flex items-center justify-center px-2">
          <AnimatePresence mode="wait">
            {subtitleVisible && currentSubtitle ? (
              <motion.p
                key={currentSubtitle}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25, ease: "easeInOut" }}
                className="text-center text-lg md:text-xl font-medium leading-snug break-words overflow-hidden w-full"
                style={{
                  color: isSpeaking
                    ? "hsl(var(--agent-text, 210 100% 80%))"
                    : "hsl(var(--user-text, 0 0% 95%))",
                }}
              >
                {currentSubtitle}
              </motion.p>
            ) : (
              <motion.p
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.4 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="text-center text-sm text-muted-foreground"
              >
                Tap the mic to speak
              </motion.p>
            )}
          </AnimatePresence>
        </div>

        {/* Waveform — active when user is listening OR Vapi is speaking */}
        <div className="w-full">
          <WaveformVisualizer
            isActive={isListening || isSpeaking}
            analyserNode={analyserNode}
          />
        </div>

        {/* Mic button + contextual label */}
        <div className="py-2 flex flex-col items-center gap-2">
          {isSubmitting ? (
            <Loader2 className="w-7 h-7 animate-spin text-primary" />
          ) : (
            <MicButton
              isListening={isListening}
              onToggle={toggleListening}
            />
          )}
          <AnimatePresence mode="wait">
            <motion.span
              key={micLabel}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 0.55, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2 }}
              className="text-xs text-muted-foreground select-none"
            >
              {micLabel}
            </motion.span>
          </AnimatePresence>
        </div>

        {/* Speaking indicator dots */}
        <AnimatePresence>
          {isSpeaking && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 6 }}
              className="flex items-center gap-2 pb-1"
            >
              <div className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: "hsl(var(--glow-primary))" }}
                    animate={{ opacity: [0.3, 1, 0.3], scaleY: [0.6, 1.4, 0.6] }}
                    transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
                  />
                ))}
              </div>
              <span className="text-xs text-muted-foreground">Jeffrey's Archive is speaking…</span>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </motion.div>
  );
};

export default VoiceAssistant;