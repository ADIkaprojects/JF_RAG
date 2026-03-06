import { motion } from "framer-motion";
import { type LucideIcon } from "lucide-react";

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  footer: string;
  delay?: number;
}

const FeatureCard = ({ icon: Icon, title, description, footer, delay = 0 }: FeatureCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ scale: 1.03, boxShadow: "0 16px 40px rgba(0,0,0,0.08)" }}
      className="w-full max-w-[280px] h-[200px] rounded-2xl bg-card p-5 shadow-card flex flex-col justify-between cursor-default"
    >
      <div>
        <div className="w-10 h-10 rounded-xl bg-secondary flex items-center justify-center mb-3">
          <Icon size={20} className="text-accent" />
        </div>
        <h3 className="font-display text-sm font-semibold mb-1 text-foreground">{title}</h3>
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">{description}</p>
      </div>
      <span className="text-[11px] text-muted-foreground font-medium">{footer}</span>
    </motion.div>
  );
};

export default FeatureCard;
