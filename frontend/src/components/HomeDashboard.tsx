import { Layers, FileStack, GitBranch, Sparkles } from "lucide-react";
import PromptInput from "./PromptInput";
import { motion } from "framer-motion";

interface HomeDashboardProps {
  onSend: (message: string, file?: File) => void;
  onVoiceOpen?: () => void;
}

const cards = [
  {
    icon: Layers,
    title: "Hybrid Retrieval",
    description: "BM25 keyword search fused with dense semantic embeddings — retrieves the most relevant chunks from your entire corpus.",
    tag: "RAG Core",
    color: "hsl(198 90% 56%)",
  },
  {
    icon: FileStack,
    title: "Multimodal Docs",
    description: "Query across PDFs, CSV datasets, and images in a single conversation. CLIP-powered visual search included.",
    tag: "Document AI",
    color: "hsl(260 60% 65%)",
  },
  {
    icon: GitBranch,
    title: "Cross-Encoder Reranking",
    description: "ms-marco MiniLM reranker ensures the highest-confidence chunks surface before generation.",
    tag: "Precision",
    color: "hsl(142 60% 50%)",
  },
];

const quickActions = [
  { label: "Search knowledge base", query: "Search my knowledge base and give me an overview" },
  { label: "Summarize PDF documents", query: "Summarize the key findings from my PDF documents" },
  { label: "Analyze CSV data", query: "Analyze patterns and insights from my CSV datasets" },
  { label: "Find relevant images", query: "Find and describe the most relevant images in my corpus" },
];

const HomeDashboard = ({ onSend, onVoiceOpen }: HomeDashboardProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-full px-8 py-6 gap-8 overflow-y-auto">

      {/* Hero text */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center max-w-2xl"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-5"
          style={{
            background: "hsl(198 90% 56% / 0.1)",
            border: "1px solid hsl(198 90% 56% / 0.2)",
            color: "hsl(198 90% 70%)",
          }}>
          <Sparkles size={11} />
          MultiModal RAG System
        </div>

        <h1 className="text-5xl font-normal leading-tight mb-4">
          <span className="text-gradient">Ask anything</span>
          <br />
          <span className="text-muted-foreground" style={{ fontFamily: "Instrument Serif, serif", fontStyle: "italic" }}>
            from your documents
          </span>
        </h1>

        <p className="text-base text-muted-foreground leading-relaxed max-w-lg mx-auto">
          Hybrid retrieval, HyDE query expansion, cross-encoder reranking —
          all working together to find the right answer.
        </p>
      </motion.div>

      {/* Feature cards */}
      <div className="flex items-stretch gap-4 flex-wrap justify-center">
        {cards.map(({ icon: Icon, title, description, tag, color }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 + i * 0.08 }}
            whileHover={{ y: -3, transition: { duration: 0.2 } }}
            className="w-[220px] rounded-2xl p-4 flex flex-col gap-3 cursor-default transition-colors duration-300"
            style={{
              background: `hsl(var(--feature-card-bg))`,
              border: `1px solid hsl(var(--feature-card-border))`,
            }}
          >
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: `${color}18` }}>
              <Icon size={18} style={{ color }} />
            </div>
            <div>
              <h3 className="text-base font-semibold text-foreground mb-1.5 tracking-wide">
                {title}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
            </div>
            <span className="text-[11px] font-medium mt-auto"
              style={{ color: `${color}99` }}>{tag}</span>
          </motion.div>
        ))}
      </div>

      {/* Prompt + quick actions */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.35 }}
        className="w-full max-w-2xl"
      >
        <PromptInput onSend={onSend} onVoiceOpen={onVoiceOpen} />
        <div className="flex flex-wrap gap-2 mt-3 justify-center">
          {quickActions.map((action) => (
            <button
              key={action.label}
              onClick={() => onSend(action.query)}
              className="text-xs rounded-full px-4 py-1.5 transition-all hover:scale-105 active:scale-95"
              style={{
                background: `hsl(var(--feature-card-bg))`,
                border: `1px solid hsl(var(--feature-card-border))`,
                color: "hsl(220 10% 65%)",
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.borderColor = "hsl(198 90% 56% / 0.4)";
                (e.currentTarget as HTMLElement).style.color = "hsl(198 90% 72%)";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.borderColor = "hsl(220 14% 20%)";
                (e.currentTarget as HTMLElement).style.color = "hsl(220 10% 65%)";
              }}
            >
              {action.label}
            </button>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default HomeDashboard;
