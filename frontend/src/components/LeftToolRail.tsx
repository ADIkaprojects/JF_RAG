import { useEffect, useRef, useState } from "react";
import {
  Plus, Search, SlidersHorizontal, Database, Activity,
  Settings, CheckCircle2, XCircle, Trash2, Loader2, Brain
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { api, type StatusResponse, type CacheStats } from "@/lib/api";
import type { Session } from "@/hooks/useConversationHistory";

interface Settings {
  topK: number;
  rerankN: number;
  provider: string;
  useCache: boolean;
  useHyde: boolean;
  useParentChild: boolean;
}

interface LeftToolRailProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  onNewChat: () => void;
  sessions: Session[];
  currentSessionId: string | null;
  onRestoreSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  settings: Settings;
  onSettingsChange: (s: Settings) => void;
}

const tools = [
  { id: "new", icon: Plus, label: "New Chat" },
  { id: "retrieval", icon: SlidersHorizontal, label: "Retrieval Settings" },
  { id: "status", icon: Activity, label: "System Status" },
  { id: "cache", icon: Database, label: "Cache" },
  { id: "settings", icon: Settings, label: "LLM Settings" },
];

const panelTools = ["retrieval", "status", "cache", "settings"];

const LeftToolRail = ({
  activeTab,
  onTabChange,
  onNewChat,
  sessions,
  currentSessionId,
  onRestoreSession,
  onDeleteSession,
  settings,
  onSettingsChange,
}: LeftToolRailProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [openPanel, setOpenPanel] = useState<string | null>(null);
  const [statusData, setStatusData] = useState<StatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [cacheLoading, setCacheLoading] = useState(false);
  const [clearLoading, setClearLoading] = useState(false);
  const [clearSuccess, setClearSuccess] = useState(false);

  useEffect(() => {
    const handleGlobalClick = (event: MouseEvent | TouchEvent) => {
      if (!openPanel) return;
      const container = containerRef.current;
      if (!container) return;
      if (!container.contains(event.target as Node)) {
        setOpenPanel(null);
      }
    };

    document.addEventListener("mousedown", handleGlobalClick);
    document.addEventListener("touchstart", handleGlobalClick);

    return () => {
      document.removeEventListener("mousedown", handleGlobalClick);
      document.removeEventListener("touchstart", handleGlobalClick);
    };
  }, [openPanel]);

  const handleClick = (id: string) => {
    if (id === "new") {
      const next = openPanel === "new" ? null : "new";
      setOpenPanel(next);
      onTabChange("new");
      return;
    }
    if (panelTools.includes(id)) {
      setOpenPanel(openPanel === id ? null : id);
      onTabChange(id);
    } else {
      onTabChange(id); setOpenPanel(null);
    }
  };

  const checkStatus = async () => {
    setStatusLoading(true);
    try {
      const data = await api.status();
      setStatusData(data);
    } catch {
      setStatusData(null);
    } finally {
      setStatusLoading(false);
    }
  };

  const fetchCacheStats = async () => {
    setCacheLoading(true);
    try {
      const data = await api.cacheStats();
      setCacheStats(data);
    } catch {
      setCacheStats(null);
    } finally {
      setCacheLoading(false);
    }
  };

  const clearCache = async () => {
    setClearLoading(true);
    try {
      await api.cacheClear();
      setClearSuccess(true);
      setCacheStats(null);
      setTimeout(() => setClearSuccess(false), 2000);
    } catch {} finally {
      setClearLoading(false);
    }
  };

  const renderPanel = (id: string) => {
    switch (id) {
      case "new":
        return (
          <div className="space-y-4">
            <PanelTitle>Conversations</PanelTitle>
            <button
              onClick={() => {
                onNewChat();
                setOpenPanel(null);
              }}
              className="w-full rounded-xl text-xs py-2.5 flex items-center justify-center gap-2 font-medium transition-all"
              style={{
                background: "hsl(var(--button-bg-active))",
                color: "hsl(var(--button-text-active))",
              }}
            >
              <Plus size={14} />
              Start New Conversation
            </button>

            <div className="pt-1">
              <p className="text-[11px] text-muted-foreground mb-2">
                Previous sessions
              </p>

              {sessions.length === 0 ? (
                <p className="text-[11px] text-muted-foreground/60 italic">
                  No previous conversations yet. Start chatting!
                </p>
              ) : (
                <div className="max-h-[400px] overflow-y-auto space-y-1.5 pr-1">
                  {sessions.slice(0, 20).map((session) => {
                    const isActive = currentSessionId === session.id;
                    return (
                      <div
                        key={session.id}
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          onRestoreSession(session.id);
                          setOpenPanel(null);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            onRestoreSession(session.id);
                            setOpenPanel(null);
                          }
                        }}
                        className="w-full flex items-start gap-2 rounded-xl px-2.5 py-2 text-left group transition-colors cursor-pointer"
                        style={{
                          background: isActive
                            ? "hsl(198 90% 56% / 0.12)"
                            : "transparent",
                          border: isActive
                            ? "1px solid hsl(198 90% 56% / 0.5)"
                            : "1px solid transparent",
                        }}
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-foreground font-medium truncate">
                            {session.title || "New Conversation"}
                          </p>
                          <p className="text-[10px] text-muted-foreground mt-0.5">
                            {formatRelativeTime(session.updatedAt)}
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 w-6 h-6 rounded-lg flex items-center justify-center text-muted-foreground/40 hover:text-destructive hover:bg-destructive/10 transition-all shrink-0"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        );

      case "settings":
        return (
          <div className="space-y-4">
            <PanelTitle>LLM Settings</PanelTitle>
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">Provider</label>
              <select
                value={settings.provider}
                onChange={e => onSettingsChange({ ...settings, provider: e.target.value })}
                className="w-full rounded-xl text-sm px-3 py-2 outline-none transition-colors duration-300"
                style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))`, color: `hsl(var(--input-text))` }}
              >
                <option value="auto">Auto (recommended)</option>
                <option value="ollama">Ollama (local)</option>
                <option value="groq">Groq (cloud)</option>
              </select>
            </div>
            <Toggle
              label="Use Cache"
              description="Skip re-querying duplicate questions"
              value={settings.useCache}
              onChange={v => onSettingsChange({ ...settings, useCache: v })}
            />
            <Toggle
              label="HyDE"
              description="Generate hypothetical doc for better retrieval"
              value={settings.useHyde}
              onChange={v => onSettingsChange({ ...settings, useHyde: v })}
            />
            <Toggle
              label="Parent-Child Retrieval"
              description="Expand child chunks to parent context"
              value={settings.useParentChild}
              onChange={v => onSettingsChange({ ...settings, useParentChild: v })}
            />
          </div>
        );

      case "retrieval":
        return (
          <div className="space-y-4">
            <PanelTitle>Retrieval Settings</PanelTitle>
            <SliderField
              label="Top K"
              description="Chunks to retrieve"
              value={settings.topK}
              min={1} max={50}
              onChange={v => onSettingsChange({ ...settings, topK: v })}
            />
            <SliderField
              label="Rerank Top N"
              description="Final results after reranking"
              value={settings.rerankN}
              min={1} max={20}
              onChange={v => onSettingsChange({ ...settings, rerankN: v })}
            />
          </div>
        );

      case "status":
        return (
          <div className="space-y-3">
            <PanelTitle>System Status</PanelTitle>
            <button
              onClick={checkStatus}
              disabled={statusLoading}
              className="w-full rounded-xl text-xs py-2 flex items-center justify-center gap-2 font-medium transition-all"
              style={{ background: "hsl(var(--button-bg-active))", color: "hsl(var(--button-text-active))" }}
            >
              {statusLoading ? <Loader2 size={13} className="animate-spin" /> : <Activity size={13} />}
              {statusLoading ? "Checking…" : "Check Status"}
            </button>

            {statusData && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2">
                {/* Health */}
                <div className="rounded-xl p-3 space-y-1.5 transition-colors duration-300"
                  style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))` }}>
                  {Object.entries(statusData.components).map(([key, val]) => (
                    <StatusRow key={key} label={key.replace(/_/g, " ")} ok={val === "active"} />
                  ))}
                </div>

                {/* Metrics */}
                <div className="rounded-xl p-3 grid grid-cols-2 gap-2 transition-colors duration-300"
                  style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))` }}>
                  <Stat label="FAISS vectors" value={statusData.vector_store.faiss_vectors} />
                  <Stat label="BM25 chunks" value={statusData.vector_store.bm25_chunks} />
                  <Stat label="Total queries" value={statusData.performance_metrics.total_queries} />
                  <Stat label="Avg time" value={`${statusData.performance_metrics.avg_execution_time}s`} />
                </div>
              </motion.div>
            )}
          </div>
        );

      case "cache":
        return (
          <div className="space-y-3">
            <PanelTitle>Cache</PanelTitle>
            <div className="flex gap-2">
              <button
                onClick={fetchCacheStats}
                disabled={cacheLoading}
                className="flex-1 rounded-xl text-xs py-2 flex items-center justify-center gap-1.5 transition-all font-medium"
                style={{ background: "hsl(var(--button-bg-inactive))", border: "1px solid hsl(var(--button-text-inactive) / 0.2)", color: "hsl(var(--button-text-inactive))" }}
              >
                {cacheLoading ? <Loader2 size={12} className="animate-spin" /> : <Database size={12} />}
                Stats
              </button>
              <button
                onClick={clearCache}
                disabled={clearLoading}
                className="flex-1 rounded-xl text-xs py-2 flex items-center justify-center gap-1.5 transition-all font-medium"
                style={{ background: clearSuccess ? "hsl(142 60% 40% / 0.2)" : "hsl(0 72% 51% / 0.15)", border: `1px solid ${clearSuccess ? "hsl(142 60% 40% / 0.3)" : "hsl(0 72% 51% / 0.3)"}`, color: clearSuccess ? "hsl(142 60% 65%)" : "hsl(0 72% 70%)" }}
              >
                {clearLoading ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                {clearSuccess ? "Cleared!" : "Clear"}
              </button>
            </div>
            {cacheStats && (
              <div className="rounded-xl p-3 transition-colors duration-300"
                style={{ background: `hsl(var(--feature-card-bg))`, border: `1px solid hsl(var(--feature-card-border))` }}>
                <p className="text-xs text-muted-foreground">Cached queries: <span className="text-foreground font-medium">{cacheStats.size}</span></p>
                <p className="text-[11px] text-muted-foreground/50 mt-1 truncate">{cacheStats.cache_dir}</p>
              </div>
            )}
          </div>
        );

      default: return null;
    }
  };

  return (
    <nav
      ref={containerRef}
      className="hidden md:flex w-[68px] flex-col items-center py-5 gap-3 glass-rail rounded-l-3xl border-r border-white/[0.04] relative z-10"
    >
      {tools.map(({ id, icon: Icon, label }) => {
        const isOpen = openPanel === id;
        return (
          <div key={id} className="relative">
            <motion.button
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.92 }}
              onClick={() => handleClick(id)}
              title={label}
              className="w-11 h-11 rounded-2xl flex items-center justify-center transition-all duration-150"
              style={{
                background: isOpen ? "hsl(var(--button-bg-active))" : "hsl(var(--button-bg-inactive))",
                border: `1px solid ${isOpen ? "transparent" : "hsl(var(--button-text-inactive) / 0.2)"}`,
                color: isOpen ? "hsl(var(--button-text-active))" : "hsl(var(--button-text-inactive))",
                boxShadow: isOpen ? "0 4px 16px hsl(198 90% 56% / 0.3)" : "none",
              }}
            >
              <Icon size={18} />
            </motion.button>

            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ opacity: 0, x: -8, scale: 0.96 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: -8, scale: 0.96 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  className="absolute left-[calc(100%+10px)] top-0 w-[280px] rounded-2xl p-4 z-50 transition-colors duration-300"
                  style={{
                    background: `hsl(var(--feature-card-bg))`,
                    border: `1px solid hsl(var(--feature-card-border))`,
                    boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
                  }}
                >
                  {renderPanel(id)}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </nav>
  );
};

// Sub-components

const PanelTitle = ({ children }: { children: React.ReactNode }) => (
  <h3 className="text-base font-semibold text-foreground tracking-wide">
    {children}
  </h3>
);

const Toggle = ({ label, description, value, onChange }: {
  label: string; description: string; value: boolean; onChange: (v: boolean) => void;
}) => (
  <div className="flex items-start justify-between gap-3">
    <div>
      <p className="text-xs text-foreground font-medium">{label}</p>
      <p className="text-[11px] text-muted-foreground leading-snug mt-0.5">{description}</p>
    </div>
    <button
      onClick={() => onChange(!value)}
      className="w-10 h-5 rounded-full flex items-center px-0.5 transition-all shrink-0 mt-0.5"
      style={{ background: value ? "hsl(var(--button-bg-active))" : "hsl(var(--button-bg-inactive))" }}
    >
      <motion.span
        layout
        className="w-4 h-4 rounded-full shadow"
        style={{ background: value ? "hsl(var(--button-text-active))" : "hsl(var(--button-text-inactive))" }}
        animate={{ x: value ? 20 : 0 }}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    </button>
  </div>
);

const SliderField = ({ label, description, value, min, max, onChange }: {
  label: string; description: string; value: number; min: number; max: number; onChange: (v: number) => void;
}) => (
  <div>
    <div className="flex justify-between items-baseline mb-1.5">
      <div>
        <span className="text-xs text-foreground font-medium">{label}</span>
        <span className="text-[11px] text-muted-foreground ml-1.5">{description}</span>
      </div>
      <span className="text-xs font-semibold text-accent">{value}</span>
    </div>
    <input
      type="range" min={min} max={max} value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
      style={{ accentColor: "hsl(198 90% 56%)" }}
    />
    <div className="flex justify-between text-[10px] text-muted-foreground/50 mt-1">
      <span>{min}</span><span>{max}</span>
    </div>
  </div>
);

const StatusRow = ({ label, ok }: { label: string; ok: boolean }) => (
  <div className="flex items-center gap-2 text-xs">
    {ok
      ? <CheckCircle2 size={13} className="text-green-500 shrink-0" />
      : <XCircle size={13} className="text-destructive shrink-0" />
    }
    <span className="text-foreground capitalize">{label}</span>
    <span className={`ml-auto text-[11px] font-medium ${ok ? "text-green-500" : "text-destructive"}`}>
      {ok ? "active" : "offline"}
    </span>
  </div>
);

const Stat = ({ label, value }: { label: string; value: string | number }) => (
  <div>
    <p className="text-[10px] text-muted-foreground">{label}</p>
    <p className="text-sm font-semibold text-foreground">{value}</p>
  </div>
);

const formatRelativeTime = (iso: string) => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";

  const diffMs = Date.now() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 60) return "Just now";
  if (diffMin < 60) return `${diffMin} min ago`;
  if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? "" : "s"} ago`;
  if (diffDay === 1) return "Yesterday";
  if (diffDay < 7) return `${diffDay} days ago`;

  return date.toLocaleDateString();
};

export default LeftToolRail;
