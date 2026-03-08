import { useState, useRef, useEffect } from "react";import { ArrowUp, Paperclip, Mic } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export interface PromptInputProps {
  onSend: (message: string, file?: File) => void;
  disabled?: boolean;
  onVoiceOpen?: () => void;  // Feature 4: opens the voice overlay
}

const PromptInput = ({ onSend, disabled, onVoiceOpen }: PromptInputProps) => {
  const [value, setValue] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [showFileMenu, setShowFileMenu] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);
  const fileMenuRef = useRef<HTMLDivElement | null>(null);

  // Feature 4: voice input only supported in Chrome, Edge, Safari
  const speechSupported =
    typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  // Close file menu when clicking outside
  useEffect(() => {
    if (!showFileMenu) return;
    const handleOutsideClick = (e: MouseEvent | TouchEvent) => {
      if (fileMenuRef.current && !fileMenuRef.current.contains(e.target as Node)) {
        setShowFileMenu(false);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("touchstart", handleOutsideClick);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("touchstart", handleOutsideClick);
    };
  }, [showFileMenu]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const handleSubmit = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim(), file ?? undefined);
    setValue("");
    setFile(null);
    setShowFileMenu(false);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;

  const handleFileSelected = (selected: File | null, input: HTMLInputElement | null) => {
    if (!input) return;
    if (!selected) {
      setFile(null);
      input.value = "";
      return;
    }

    const allowed = ["application/pdf", "image/png", "image/jpeg", "image/webp"];
    if (!allowed.includes(selected.type) || selected.size > 50 * 1024 * 1024) {
      alert("Invalid file type or file too large (max 50MB). Allowed: PDF, PNG, JPG, JPEG, WEBP.");
      input.value = "";
      setFile(null);
      return;
    }

    setFile(selected);
  };

  return (
    <div className="relative rounded-2xl transition-all duration-200"
      style={{
        background: `hsl(var(--feature-card-bg))`,
        border: `1px solid hsl(var(--feature-card-border))`,
        boxShadow: "0 0 0 0 transparent",
      }}
      onFocusCapture={e => {
        (e.currentTarget as HTMLElement).style.borderColor = "hsl(198 90% 56% / 0.4)";
        (e.currentTarget as HTMLElement).style.boxShadow = "0 0 0 3px hsl(198 90% 56% / 0.08)";
      }}
      onBlurCapture={e => {
        (e.currentTarget as HTMLElement).style.borderColor = `hsl(var(--feature-card-border))`;
        (e.currentTarget as HTMLElement).style.boxShadow = "none";
      }}
    >
      <div className="flex items-end gap-3 px-4 py-3">
        <div className="relative" ref={fileMenuRef}>
          <button
            type="button"
            className="mb-0.5 w-8 h-8 rounded-xl flex items-center justify-center transition-colors shrink-0"
            style={{ color: "hsl(var(--button-text-inactive))" }}
            title="Attach file"
            disabled={disabled}
            onClick={() => setShowFileMenu(prev => !prev)}
            onMouseEnter={e => (e.currentTarget.style.color = "hsl(var(--button-text-inactive) / 0.8)")}
            onMouseLeave={e => (e.currentTarget.style.color = "hsl(var(--button-text-inactive))")}
          >
            <Paperclip size={16} />
          </button>

          <AnimatePresence>
            {showFileMenu && !disabled && (
              <motion.div
                initial={{ opacity: 0, y: 4, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 4, scale: 0.98 }}
                transition={{ duration: 0.15 }}
                className="absolute left-0 bottom-[110%] w-40 rounded-xl p-2 z-50"
                style={{
                  background: `hsl(var(--feature-card-bg))`,
                  border: `1px solid hsl(var(--feature-card-border))`,
                  boxShadow: "0 12px 32px rgba(0,0,0,0.5)",
                }}
              >
                <p className="text-[11px] text-muted-foreground mb-1.5 px-1">
                  Attach as
                </p>
                <button
                  type="button"
                  className="w-full text-xs px-2 py-1.5 rounded-lg flex items-center justify-between hover:bg-white/5 transition-colors"
                  onClick={() => {
                    setShowFileMenu(false);
                    imageInputRef.current?.click();
                  }}
                >
                  <span>Image</span>
                  <span className="text-[10px] text-muted-foreground/70">PNG/JPEG</span>
                </button>
                <button
                  type="button"
                  className="mt-1 w-full text-xs px-2 py-1.5 rounded-lg flex items-center justify-between hover:bg-white/5 transition-colors"
                  onClick={() => {
                    setShowFileMenu(false);
                    pdfInputRef.current?.click();
                  }}
                >
                  <span>PDF</span>
                  <span className="text-[10px] text-muted-foreground/70">.pdf</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <input
            ref={imageInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.webp"
            style={{ display: "none" }}
            disabled={disabled}
            onChange={e => {
              const selected = e.target.files?.[0] ?? null;
              handleFileSelected(selected, imageInputRef.current);
            }}
          />
          <input
            ref={pdfInputRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            disabled={disabled}
            onChange={e => {
              const selected = e.target.files?.[0] ?? null;
              handleFileSelected(selected, pdfInputRef.current);
            }}
          />
        </div>

        {/* Feature 4: voice input button — only shown in supported browsers */}
        {speechSupported && onVoiceOpen && (
          <motion.button
            type="button"
            whileTap={{ scale: 0.9 }}
            onClick={onVoiceOpen}
            disabled={disabled}
            className="mb-0.5 w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0"
            style={{ color: "hsl(198 90% 60%)" }}
            title="Voice query (Chrome / Safari)"
            aria-label="Start voice input"
            onMouseEnter={e => (e.currentTarget.style.color = "hsl(198 90% 80%)")}
            onMouseLeave={e => (e.currentTarget.style.color = "hsl(198 90% 60%)")}
          >
            <Mic size={16} />
          </motion.button>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your documents…"
          rows={1}
          disabled={disabled}
          className="flex-1 bg-transparent outline-none resize-none text-sm text-foreground placeholder:text-muted-foreground/50 leading-relaxed"
          style={{ minHeight: "24px", maxHeight: "160px" }}
        />

        <AnimatePresence>
          <motion.button
            onClick={handleSubmit}
            disabled={!canSend}
            whileHover={canSend ? { scale: 1.05 } : {}}
            whileTap={canSend ? { scale: 0.95 } : {}}
            className="mb-0.5 w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-all duration-200"
            style={{
              background: canSend ? "hsl(var(--button-bg-active))" : "hsl(var(--button-bg-inactive))",
              color: canSend ? "hsl(var(--button-text-active))" : "hsl(var(--button-text-inactive))",
            }}
            title="Send (Enter)"
          >
            <ArrowUp size={15} strokeWidth={2.5} />
          </motion.button>
        </AnimatePresence>
      </div>

      <div className="px-4 pb-2 flex items-center justify-between">
        <span className="hidden sm:inline text-[10px] text-muted-foreground/30">
          Enter to send · Shift+Enter for newline
        </span>
        {file && (
          <span className="ml-2 px-2 py-0.5 rounded-full bg-accent/10 text-xs text-accent border border-accent/20">
            {file.name}
          </span>
        )}
        {value.length > 0 && (
          <span className="text-[10px] text-muted-foreground/30">{value.length}</span>
        )}
      </div>
    </div>
  );
};

export default PromptInput;
