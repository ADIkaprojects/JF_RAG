import { useState, useCallback, useRef, useEffect } from "react";
import { Moon, Sun } from "lucide-react";
import { AnimatePresence } from "framer-motion";
import LeftToolRail from "@/components/LeftToolRail";
import HomeDashboard from "@/components/HomeDashboard";
import ChatView, { type ChatMessage } from "@/components/ChatView";
import VoiceOverlay from "@/components/VoiceOverlay";
import { api, isImageQuery, type QueryRequest } from "@/lib/api";
import { useTheme } from "@/hooks/useTheme";
import { useConversationHistory, type Session } from "@/hooks/useConversationHistory";
import { useLikedMessages } from "@/hooks/useLikedMessages";

const Index = () => {
  const { theme, toggleTheme } = useTheme();
  const [activeTab, setActiveTab] = useState("search");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [view, setView] = useState<"home" | "chat">("home");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const {
    sessions,
    saveSession,
    loadSession,
    deleteSession,
    createNewSession,
  } = useConversationHistory();

  const { likedMessages, addLike, removeLike } = useLikedMessages();

  // Track which assistant message IDs the user has liked (for ThumbsUp highlight)
  const [likedMsgIds, setLikedMsgIds] = useState<Set<string>>(new Set());

  // Feature 4: voice overlay
  const [voiceOverlayOpen, setVoiceOverlayOpen] = useState(false);

  // Settings state — lifted so rail can control, chat can read
  const [settings, setSettings] = useState<{
    topK: number;
    rerankN: number;
    provider: string;
    useCache: boolean;
    useHyde: boolean;
    useParentChild: boolean;
  }>({
    topK: 10,
    rerankN: 5,
    provider: "auto",
    useCache: true,
    useHyde: true,
    // Parent-Child Retrieval disabled by default (toggle OFF)
    useParentChild: false,
  });

  const handleSend = useCallback(async (text: string, file?: File) => {
    const visualQuery = isImageQuery(text);

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      attachedFileName: file?.name,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setView("chat");
    setIsLoading(true);

    try {
      let data;
      if (file) {
        // Build FormData for file upload
        const formData = new FormData();
        formData.append("query", text);
        formData.append("top_k", String(settings.topK));
        formData.append("rerank_top_n", String(settings.rerankN));
        // When UI is set to Auto, explicitly send 'auto' so backend can treat
        // it as "always use Groq".
        formData.append(
          "force_provider",
          settings.provider === "auto" ? "auto" : settings.provider
        );
        formData.append("use_cache", "false"); // File uploads are never cached
        formData.append("use_hyde", String(settings.useHyde));
        formData.append("use_parent_child", String(settings.useParentChild));
        formData.append("use_image_search", "true");
        formData.append("file", file);
        data = await api.queryWithFile(formData);
      } else {
        const req: QueryRequest = {
          query: text,
          top_k: settings.topK,
          rerank_top_n: settings.rerankN,
          // Send 'auto' to backend so it can route to Groq explicitly.
          force_provider: settings.provider === "auto" ? "auto" : settings.provider,
          use_cache: settings.useCache,
          use_hyde: settings.useHyde,
          use_parent_child: settings.useParentChild,
          use_image_search: true,
        };
        data = await api.query(req);
      }

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: data.sources.map((s: any) => ({
          name: s.source_file,
          page: s.page,
          snippet: s.text,
          docType: s.doc_type,
          confidence: s.confidence_score,
        })),
        imageSources: data.image_sources,
        imageQueryDetected: visualQuery,
        metadata: {
          provider: data.provider,
          model: data.model,
          cached: data.cached,
          hydeUsed: data.hyde_used,
          rewrittenQuery: data.rewritten_query,
          guardrailTriggered: data.guardrail_triggered,
        },
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `⚠️ **Error:** ${err.message ?? "Failed to connect to API. Make sure the backend is running at \`http://localhost:8000\`."}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  const handleRetry = useCallback(async (userQuery: string) => {
    if (!userQuery) return;
    try { await api.cacheDeleteEntry(userQuery); } catch { /* ignore */ }
    handleSend(userQuery);
  }, [handleSend]);

  const handleLike = useCallback((msg: ChatMessage, userQuery: string) => {
    if (likedMsgIds.has(msg.id)) {
      // Toggle off — remove from liked
      const existing = likedMessages.find(m => m.query === userQuery);
      if (existing) removeLike(existing.id);
      setLikedMsgIds((prev) => { const next = new Set(prev); next.delete(msg.id); return next; });
    } else {
      addLike({ query: userQuery, answer: msg.content, sources: msg.sources });
      setLikedMsgIds((prev) => new Set(prev).add(msg.id));
    }
  }, [addLike, likedMsgIds, likedMessages, removeLike]);

  const handleDislike = useCallback(async (userQuery: string) => {
    if (!userQuery) return;
    // Also remove from liked messages if present
    const existing = likedMessages.find(m => m.query === userQuery);
    if (existing) removeLike(existing.id);
    try { await api.cacheDeleteEntry(userQuery); } catch { /* ignore */ }
  }, [likedMessages, removeLike]);

  // Initialize a fresh session on first mount
  useEffect(() => {
    const id = createNewSession();
    if (id) {
      setCurrentSessionId(id);
    }
  }, [createNewSession]);

  // Auto-save conversation whenever messages change
  useEffect(() => {
    if (!currentSessionId) return;

    const session: Session = {
      id: currentSessionId,
      title: "",
      createdAt: "",
      updatedAt: "",
      messages,
    };

    saveSession(session);
  }, [messages, currentSessionId, saveSession]);

  const handleNewChat = useCallback(() => {
    const id = createNewSession();
    if (!id) return;
    setCurrentSessionId(id);
    setMessages([]);
    setView("home");
  }, [createNewSession]);

  const handleRestoreSession = useCallback(
    (sessionId: string) => {
      const session = loadSession(sessionId);
      if (!session) return;
      setCurrentSessionId(sessionId);
      setMessages(session.messages || []);
      setView("chat");
    },
    [loadSession]
  );

  const handleDeleteSession = useCallback(
    (sessionId: string) => {
      deleteSession(sessionId);
      if (sessionId === currentSessionId) {
        const id = createNewSession();
        if (!id) return;
        setCurrentSessionId(id);
        setMessages([]);
        setView("home");
      }
    },
    [currentSessionId, createNewSession, deleteSession]
  );

  // Feature 4: voice overlay handlers
  const handleVoiceOpen = useCallback(() => setVoiceOverlayOpen(true), []);
  const handleVoiceClose = useCallback(() => {
    setVoiceOverlayOpen(false);
  }, []);

  // Feature 4: voice query handler used by VoiceAssistant
  // Returns the assistant answer so the voice overlay can display and speak it.
  const handleVoiceQuery = useCallback(async (transcript: string) => {
    const trimmed = transcript.trim();
    if (!trimmed) return "";

    const visualQuery = isImageQuery(trimmed);

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setView("chat");
    setIsLoading(true);

    try {
      const req: QueryRequest = {
        query: trimmed,
        top_k: settings.topK,
        rerank_top_n: settings.rerankN,
        force_provider: settings.provider === "auto" ? "auto" : settings.provider,
        use_cache: settings.useCache,
        use_hyde: settings.useHyde,
        use_parent_child: settings.useParentChild,
        use_image_search: true,
      };

      const data = await api.query(req);

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: data.sources.map((s: any) => ({
          name: s.source_file,
          page: s.page,
          snippet: s.text,
          docType: s.doc_type,
          confidence: s.confidence_score,
        })),
        imageSources: data.image_sources,
        imageQueryDetected: visualQuery,
        metadata: {
          provider: data.provider,
          model: data.model,
          cached: data.cached,
          hydeUsed: data.hyde_used,
          rewrittenQuery: data.rewritten_query,
          guardrailTriggered: data.guardrail_triggered,
        },
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      return data.answer ?? "";
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `⚠️ **Error:** ${err.message ?? "Failed to connect to API."}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
      return "";
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  return (
    <div className="min-h-screen page-gradient flex md:items-center md:justify-center md:p-3 md:p-5">
      {/* Ambient background orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-20%] left-[10%] w-[600px] h-[600px] rounded-full"
          style={{ background: "radial-gradient(circle, hsl(198 90% 56% / 0.06) 0%, transparent 70%)" }} />
        <div className="absolute bottom-[-10%] right-[5%] w-[500px] h-[500px] rounded-full"
          style={{ background: "radial-gradient(circle, hsl(260 60% 50% / 0.05) 0%, transparent 70%)" }} />
      </div>

      {/* Main container */}
      <div className="relative w-full md:max-w-[1440px] h-[100dvh] md:h-[min(92vh,900px)] glass-container md:rounded-3xl flex overflow-hidden">

        {/* Subtle top border glow */}
        <div className="absolute top-0 left-[20%] right-[20%] h-px"
          style={{ background: "linear-gradient(90deg, transparent, hsl(198 90% 56% / 0.3), transparent)" }} />

        {/* Header bar */}
        <div className="absolute top-0 left-0 right-0 h-12 flex items-center justify-between z-20 px-4 md:px-6">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/80 ml-4 md:ml-32">
            <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span className="text-xs font-medium text-muted-foreground tracking-wide">
              {view === "chat" ? "MultiModal RAG" : "MultiModal RAG — Ready"}
            </span>
          </div>
          
          {/* Theme Toggle Button - Top Right */}
          <button
            onClick={toggleTheme}
            className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:bg-accent/10"
            style={{
              background: theme === 'dark'
                ? "hsl(var(--button-bg-inactive))"
                : "hsl(220 14% 91%)",
              border: theme === 'dark'
                ? "1px solid hsl(var(--button-text-inactive) / 0.2)"
                : "1px solid hsl(220 8% 80%)",
            }}
            title={theme === 'dark' ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === 'dark' ? (
              <Sun size={16} className="text-accent" />
            ) : (
              <Moon size={16} style={{ color: "hsl(220 8% 40%)" }} />
            )}
          </button>
        </div>

        {/* Left rail */}
        <LeftToolRail
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onNewChat={handleNewChat}
          sessions={sessions}
          currentSessionId={currentSessionId}
          onRestoreSession={handleRestoreSession}
          onDeleteSession={handleDeleteSession}
          settings={settings}
          onSettingsChange={setSettings}
          likedMessages={likedMessages}
          onRemoveLike={removeLike}
        />

        {/* Main workspace */}
        <main className="flex-1 min-w-0 flex flex-col pt-12 pb-16 md:pb-0">
          {view === "home" ? (
            <HomeDashboard onSend={handleSend} onVoiceOpen={handleVoiceOpen} />
          ) : (
            <ChatView
              messages={messages}
              onSend={handleSend}
              isLoading={isLoading}
              onVoiceOpen={handleVoiceOpen}
              onRetry={handleRetry}
              onLike={handleLike}
              onDislike={handleDislike}
              likedIds={likedMsgIds}
            />
          )}
        </main>
      </div>

      {/* Feature 4: voice overlay — rendered above all other content */}
      <AnimatePresence>
        {voiceOverlayOpen && (
          <VoiceOverlay
            onClose={handleVoiceClose}
            onQueryReady={handleVoiceQuery}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default Index;
