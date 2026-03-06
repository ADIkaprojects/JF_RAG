/**
 * useVapi — Feature 4 Enhancement (v2)
 *
 * Runs Vapi as a full conversational assistant call.
 * - On startCall(): opens the session → Vapi greets the user automatically
 *   via the assistant's firstMessage ("Hey, I'm Jeffrey's Archive…")
 * - speakText(text): injects the RAG answer as an assistant message so Vapi
 *   reads it aloud in Raquel's voice
 * - onWord callback: fires with each spoken word so VoiceAssistant can drive
 *   the subtitle display one line at a time
 * - stopCall(): tears down the session cleanly
 */

import { useRef, useState, useEffect, useCallback } from "react";
import Vapi from "@vapi-ai/web";

const ASSISTANT_ID = import.meta.env.VITE_VAPI_ASSISTANT_ID as string | undefined;
const PUBLIC_KEY = import.meta.env.VITE_VAPI_PUBLIC_KEY as string | undefined;

// How many words to show per subtitle line
const WORDS_PER_LINE = 8;

interface UseVapiOptions {
  /** Called with each subtitle line as Vapi speaks */
  onSubtitleLine?: (line: string) => void;
  /** Called when Vapi finishes speaking a response */
  onSpeechEnd?: () => void;
}

export function useVapi(options: UseVapiOptions = {}) {
  const { onSubtitleLine, onSpeechEnd } = options;

  const vapiRef = useRef<Vapi | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isCallActive, setIsCallActive] = useState(false);
  // Ref mirrors isCallActive so speakText never reads stale state
  const isCallActiveRef = useRef(false);
  const [vapiError, setVapiError] = useState<string | null>(null);

  // Word buffer for building subtitle lines
  const wordBufferRef = useRef<string[]>([]);
  const transcriptBufferRef = useRef<string>("");

  // Build and emit subtitle lines from streamed transcript
  const flushSubtitleLine = useCallback(
    (force = false) => {
      const words = wordBufferRef.current;
      if (words.length === 0) return;
      if (force || words.length >= WORDS_PER_LINE) {
        const line = words.join(" ");
        onSubtitleLine?.(line);
        wordBufferRef.current = [];
      }
    },
    [onSubtitleLine]
  );

  // Initialise Vapi once
  useEffect(() => {
    if (!PUBLIC_KEY) {
      console.warn("useVapi: VITE_VAPI_PUBLIC_KEY is not set.");
      return;
    }
    if (!ASSISTANT_ID) {
      console.warn("useVapi: VITE_VAPI_ASSISTANT_ID is not set.");
      return;
    }

    const instance = new Vapi(PUBLIC_KEY);

    // Vapi streams back assistant transcript word by word via "message" events
    instance.on("message", (msg: any) => {
      // transcript-update carries partial assistant speech text
      if (msg?.type === "transcript" && msg?.role === "assistant") {
        const text: string = msg.transcript ?? "";
        // Accumulate new words since last update
        const prev = transcriptBufferRef.current;
        const newPart = text.slice(prev.length).trim();
        transcriptBufferRef.current = text;

        if (newPart) {
          const incoming = newPart.split(/\s+/).filter(Boolean);
          wordBufferRef.current.push(...incoming);
          flushSubtitleLine();
        }
      }
    });

    instance.on("speech-start", () => {
      setIsSpeaking(true);
      wordBufferRef.current = [];
      transcriptBufferRef.current = "";
    });

    instance.on("speech-end", () => {
      // Flush any remaining words as the last subtitle line
      flushSubtitleLine(true);
      wordBufferRef.current = [];
      transcriptBufferRef.current = "";
      setIsSpeaking(false);
      onSpeechEnd?.();
    });

    instance.on("call-end", () => {
      isCallActiveRef.current = false;
      setIsCallActive(false);
      setIsSpeaking(false);
    });

    instance.on("call-start-failed", (event: any) => {
      console.error("Vapi call-start-failed:", event);
      isCallActiveRef.current = false;
      setIsCallActive(false);
      setVapiError(event?.error ?? "Vapi call failed to start");
    });

    instance.on("error", (err: any) => {
      console.error("Vapi error:", err);
      setIsSpeaking(false);
      isCallActiveRef.current = false;
      setIsCallActive(false);
      setVapiError(err?.message ?? "Vapi error");
    });

    vapiRef.current = instance;

    return () => {
      instance.stop();
      vapiRef.current = null;
    };
  }, [flushSubtitleLine, onSpeechEnd]);

  /**
   * startCall — opens a Vapi session. The assistant will speak its
   * firstMessage greeting automatically ("Hey, I'm Jeffrey's Archive…").
   */
  const startCall = useCallback(async () => {
    const vapi = vapiRef.current;
    if (!vapi) return;
    if (!ASSISTANT_ID) {
      setVapiError("Vapi assistant ID is not configured. Check VITE_VAPI_ASSISTANT_ID in .env and restart the dev server.");
      return;
    }

    try {
      setVapiError(null);
      wordBufferRef.current = [];
      transcriptBufferRef.current = "";

      await vapi.start(ASSISTANT_ID);
      isCallActiveRef.current = true;
      setIsCallActive(true);
    } catch (err: any) {
      console.error("useVapi startCall error:", err);
      setVapiError(err?.message ?? "Failed to start Vapi call");
    }
  }, []);

  /**
   * speakText — injects the RAG answer as an assistant message.
   * Vapi will speak it in Raquel's voice, streaming transcript events
   * back for subtitle rendering.
   */
  const speakText = useCallback(async (text: string) => {
    const vapi = vapiRef.current;
    if (!vapi || !text.trim()) return;

    if (!ASSISTANT_ID) {
      setVapiError("Vapi assistant ID is not configured. Check VITE_VAPI_ASSISTANT_ID in .env and restart the dev server.");
      return;
    }

    try {
      wordBufferRef.current = [];
      transcriptBufferRef.current = "";

      // Use ref (not state) so we never read a stale isCallActive value
      if (!isCallActiveRef.current) {
        await vapi.start(ASSISTANT_ID);
        isCallActiveRef.current = true;
        setIsCallActive(true);
        // Small delay to let the session stabilise
        await new Promise((res) => setTimeout(res, 600));
      }

      // say() is the correct method to trigger TTS — add-message only logs to transcript
      vapi.say(text);
    } catch (err: any) {
      console.error("useVapi speakText error:", err);
      setVapiError(err?.message ?? "Failed to speak text");
    }
  }, []);

  /**
   * stopCall — cleanly ends the Vapi session.
   */
  const stopCall = useCallback(() => {
    flushSubtitleLine(true);
    vapiRef.current?.stop();
    isCallActiveRef.current = false;
    setIsCallActive(false);
    setIsSpeaking(false);
    wordBufferRef.current = [];
    transcriptBufferRef.current = "";
  }, [flushSubtitleLine]);

  return {
    startCall,
    speakText,
    stopCall,
    isSpeaking,
    isCallActive,
    vapiError,
    // Legacy alias so any other callsite using stopSpeaking still works
    stopSpeaking: stopCall,
  };
}