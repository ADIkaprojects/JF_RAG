import { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ThumbsUp, ThumbsDown, RotateCcw, FileText, Zap, Brain, Shield, Image as ImageIcon } from "lucide-react";
import PromptInput from "./PromptInput";
import ImageResultCards from "./ImageResultCards";
import type { RetrievalResult } from "@/lib/api";
import ReactMarkdown from "react-markdown";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: { name: string; page: number; snippet: string; docType?: string; confidence?: number }[];
  imageSources?: RetrievalResult[] | null;
  /** True when the originating user query contained a visual-intent keyword.
   *  Controls whether <ImageResultCards /> is rendered above this message. */
  imageQueryDetected?: boolean;
  attachedFileName?: string;
  metadata?: {
    provider: string;
    model: string;
    cached: boolean;
    hydeUsed: boolean;
    rewrittenQuery?: string;
    guardrailTriggered: boolean;
  };
  timestamp: Date;
  isError?: boolean;
}

interface ChatViewProps {
  messages: ChatMessage[];
  onSend: (message: string, file?: File) => void;
  isLoading: boolean;
  onVoiceOpen?: () => void;  // Feature 4: passed through to PromptInput
  onRetry?: (userQuery: string) => void;
  onLike?: (msg: ChatMessage, userQuery: string) => void;
  onDislike?: (userQuery: string) => void;
  likedIds?: Set<string>;
}

const ProviderBadge = ({ provider, model, cached, hydeUsed }: {
  provider: string; model: string; cached: boolean; hydeUsed: boolean;
}) => (
  <div className="flex items-center gap-2 flex-wrap mt-3 pt-3 border-t border-white/5">
    <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-medium"
      style={{ background: "hsl(198 90% 56% / 0.12)", color: "hsl(198 90% 70%)" }}>
      <Zap size={9} />
      {provider.toUpperCase()}
    </span>
    <span className="text-[10px] text-muted-foreground font-mono truncate max-w-[120px]">
      {model.split("/").pop()}
    </span>
    {cached && (
      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
        style={{ background: "hsl(142 60% 40% / 0.15)", color: "hsl(142 60% 60%)" }}>
        ⚡ cached
      </span>
    )}
    {hydeUsed && (
      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
        style={{ background: "hsl(260 60% 50% / 0.15)", color: "hsl(260 60% 72%)" }}>
        <Brain size={9} /> HyDE
      </span>
    )}
  </div>
);

const SourceCard = ({ src, index }: { src: ChatMessage["sources"][0]; index: number }) => (
  <motion.div
    initial={{ opacity: 0, y: 4 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.05 }}
    className="rounded-xl p-3 text-xs transition-colors duration-300"
    style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))` }}
  >
    <div className="flex items-start gap-2">
      <FileText size={12} className="mt-0.5 shrink-0 text-accent" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-semibold text-foreground truncate">{src.name}</span>
          <span className="text-muted-foreground shrink-0">p.{src.page}</span>
          {src.docType && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-md shrink-0 transition-colors duration-300"
              style={{ background: `hsl(var(--feature-card-border))`, color: "hsl(220 10% 55%)" }}>
              {src.docType}
            </span>
          )}
        </div>
        <p className="text-muted-foreground leading-relaxed line-clamp-2">{src.snippet}</p>
        {src.confidence !== undefined && src.confidence > 0 && (
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
              <div className="h-full rounded-full bg-accent/50"
                style={{ width: `${Math.min(src.confidence * 100, 100)}%` }} />
            </div>
            <span className="text-[10px] text-muted-foreground">{(src.confidence * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </div>
  </motion.div>
);

const ThinkingDots = () => (
  <div className="flex justify-start">
    <div className="flex items-center gap-3 px-5 py-4 rounded-2xl rounded-tl-sm transition-colors duration-300"
      style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))` }}>
      <div className="flex gap-1.5">
        {[0, 150, 300].map((delay, i) => (
          <span key={i} className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce"
            style={{ animationDelay: `${delay}ms` }} />
        ))}
      </div>
      <span className="text-xs text-muted-foreground">Retrieving and generating…</span>
    </div>
  </div>
);

const ChatView = ({ messages, onSend, isLoading, onVoiceOpen, onRetry, onLike, onDislike, likedIds }: ChatViewProps) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [expandedSources, setExpandedSources] = useState<string | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-5">
        {messages.map((msg, msgIdx) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "user" ? (
              <div className="max-w-[85%] sm:max-w-[70%] px-4 sm:px-5 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed font-medium flex items-center gap-2"
                style={{
                  background: "hsl(198 90% 56%)",
                  color: "hsl(220 16% 7%)",
                }}>
                {msg.content}
                {msg.attachedFileName && (
                  <span className="ml-2 px-2 py-0.5 rounded-full bg-accent/10 text-xs text-accent border border-accent/20">
                    {msg.attachedFileName}
                  </span>
                )}
              </div>
            ) : (
              <div className="max-w-[92%] sm:max-w-[80%] min-w-0">
                {/* Guardrail notice */}
                {msg.metadata?.guardrailTriggered && (
                  <div className="flex items-center gap-2 mb-2 text-xs px-3 py-1.5 rounded-lg"
                    style={{ background: "hsl(0 72% 51% / 0.1)", color: "hsl(0 72% 70%)", border: "1px solid hsl(0 72% 51% / 0.2)" }}>
                    <Shield size={12} />
                    Guardrail triggered
                  </div>
                )}

                {/* Rewritten query notice */}
                {msg.metadata?.rewrittenQuery && (
                  <div className="text-[11px] text-muted-foreground mb-2 flex items-center gap-1.5">
                    <Brain size={10} className="text-accent/60" />
                    Searched as: <span className="italic text-accent/80">{msg.metadata.rewrittenQuery}</span>
                  </div>
                )}

                {/* ── IMAGE RESULT CARDS (Feature 1) ──────────────────────
                    Rendered above the text bubble only when the user's query
                    contained a visual-intent keyword AND the backend returned
                    at least one image source.                              */}
                {msg.imageQueryDetected && 
                 Array.isArray(msg.imageSources) && 
                 msg.imageSources.length > 0 && (
                  <ImageResultCards sources={msg.imageSources} />
                )}

                {/* Message bubble */}
                <div className={`px-5 py-4 rounded-2xl rounded-tl-sm text-sm leading-relaxed transition-colors duration-300 ${
                  msg.isError ? "border border-destructive/30" : ""
                }`}
                  style={{
                    background: msg.isError ? "hsl(0 72% 51% / 0.08)" : `hsl(var(--feature-card-bg))`,
                    border: msg.isError ? "1px solid hsl(0 72% 51% / 0.3)" : `1px solid hsl(var(--feature-card-border))`,
                  }}>
                  <div className="prose prose-sm max-w-none dark:prose-invert
                    prose-p:text-foreground/90 prose-p:leading-relaxed
                    prose-strong:text-foreground prose-strong:font-semibold
                    prose-code:text-accent prose-code:bg-white/5 prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                    prose-pre:bg-black/30 prose-pre:border prose-pre:border-white/10">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4">
                      <button
                        onClick={() => setExpandedSources(expandedSources === msg.id ? null : msg.id)}
                        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors group"
                      >
                        <FileText size={12} className="text-accent group-hover:text-accent" />
                        <span>{msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}</span>
                        <ChevronDown size={12} className={`transition-transform ${expandedSources === msg.id ? "rotate-180" : ""}`} />
                      </button>
                      <AnimatePresence>
                        {expandedSources === msg.id && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="overflow-hidden mt-2 space-y-2"
                          >
                            {msg.sources.map((src, i) => (
                              <SourceCard key={i} src={src} index={i} />
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  {/* Image sources */}
                  {msg.imageSources && msg.imageSources.length > 0 && (
                    <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
                      <ImageIcon size={12} className="text-accent/60" />
                      {msg.imageSources.length} image result{msg.imageSources.length > 1 ? "s" : ""} used
                    </div>
                  )}

                  {/* Metadata */}
                  {msg.metadata && !msg.isError && (
                    <ProviderBadge
                      provider={msg.metadata.provider}
                      model={msg.metadata.model}
                      cached={msg.metadata.cached}
                      hydeUsed={msg.metadata.hydeUsed}
                    />
                  )}
                </div>

                {/* Action buttons */}
                <div className="flex items-center gap-2 mt-1.5 px-1">
                  <span className="text-[10px] text-muted-foreground/50">
                    {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                  <div className="flex gap-1 ml-auto">
                    {(() => {
                      const userQuery = messages.slice(0, msgIdx).reverse().find(m => m.role === "user")?.content ?? "";
                      const isLiked = likedIds?.has(msg.id) ?? false;
                      return (
                        <>
                          <button
                            title="Retry"
                            onClick={() => onRetry?.(userQuery)}
                            className="w-6 h-6 rounded-lg flex items-center justify-center text-muted-foreground/40 hover:text-muted-foreground hover:bg-white/5 transition-all"
                          >
                            <RotateCcw size={11} />
                          </button>
                          <button
                            title={isLiked ? "Unlike" : "Like"}
                            onClick={() => onLike?.(msg, userQuery)}
                            className="w-6 h-6 rounded-lg flex items-center justify-center transition-all hover:bg-white/5"
                            style={{ color: isLiked ? "hsl(142 60% 60%)" : undefined }}
                          >
                            <ThumbsUp size={11} className={isLiked ? "" : "text-muted-foreground/40 hover:text-muted-foreground"} />
                          </button>
                          <button
                            title="Dislike (removes from cache)"
                            onClick={() => onDislike?.(userQuery)}
                            className="w-6 h-6 rounded-lg flex items-center justify-center text-muted-foreground/40 hover:text-muted-foreground hover:bg-white/5 transition-all"
                          >
                            <ThumbsDown size={11} />
                          </button>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        ))}

        {isLoading && <ThinkingDots />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-3 sm:px-6 pb-5 pt-2">
        <PromptInput
          onSend={onSend}
          disabled={isLoading}
          onVoiceOpen={onVoiceOpen}
        />
      </div>
    </div>
  );
};

export default ChatView;
