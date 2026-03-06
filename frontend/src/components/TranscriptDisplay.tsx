import { motion, AnimatePresence } from "framer-motion";

export interface TranscriptEntry {
  id: string;
  role: "user" | "agent";
  text: string;
}

interface TranscriptDisplayProps {
  entries: TranscriptEntry[];
  isListening: boolean;
}

const TranscriptDisplay = ({ entries, isListening }: TranscriptDisplayProps) => {
  const lastEntry = entries[entries.length - 1];

  return (
    <div className="min-h-[120px] w-full flex flex-col justify-end px-2 overflow-hidden">
      <AnimatePresence mode="wait">
        {lastEntry ? (
          <motion.div
            key={lastEntry.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <p
              className={`text-xl md:text-2xl font-medium leading-relaxed break-words overflow-hidden w-full ${
                lastEntry.role === "user" ? "text-user-text" : "text-agent-text"
              }`}
            >
              {lastEntry.text}
              {lastEntry.role === "user" && isListening && (
                <motion.span
                  animate={{ opacity: [1, 0.3] }}
                  transition={{ duration: 0.6, repeat: Infinity, repeatType: "reverse" }}
                  className="text-primary"
                >
                  ••
                </motion.span>
              )}
            </p>
          </motion.div>
        ) : (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-xl md:text-2xl font-medium text-muted-foreground leading-relaxed"
          >
            Tap the mic to start speaking...
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
};

export default TranscriptDisplay;

