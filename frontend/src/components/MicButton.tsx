import { Mic, MicOff } from "lucide-react";
import { motion } from "framer-motion";

interface MicButtonProps {
  isListening: boolean;
  onToggle: () => void;
}

const MicButton = ({ isListening, onToggle }: MicButtonProps) => {
  return (
    <div className="relative flex items-center justify-center">
      {/* Pulse rings when active */}
      {isListening && (
        <>
          <motion.div
            className="absolute w-16 h-16 rounded-full bg-primary/20"
            animate={{ scale: [1, 2], opacity: [0.4, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
          />
          <motion.div
            className="absolute w-16 h-16 rounded-full bg-primary/15"
            animate={{ scale: [1, 2.5], opacity: [0.3, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut", delay: 0.3 }}
          />
        </>
      )}

      <motion.button
        whileTap={{ scale: 0.92 }}
        whileHover={{ scale: 1.05 }}
        onClick={onToggle}
        className={`relative z-10 w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${
          isListening
            ? "bg-foreground mic-glow-active"
            : "bg-foreground/90 mic-glow"
        }`}
      >
        {isListening ? (
          <MicOff className="w-6 h-6 text-background" />
        ) : (
          <Mic className="w-6 h-6 text-background" />
        )}
      </motion.button>
    </div>
  );
};

export default MicButton;

