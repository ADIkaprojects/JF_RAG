import { useState, useCallback } from "react";

export interface LikedMessage {
  id: string;
  query: string;
  answer: string;
  sources?: { name: string; page: number; snippet: string; docType?: string; confidence?: number }[];
  timestamp: string;
}

const STORAGE_KEY = "rag_liked_messages";

function loadFromStorage(): LikedMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function useLikedMessages() {
  const [likedMessages, setLikedMessages] = useState<LikedMessage[]>(loadFromStorage);

  const addLike = useCallback((item: Omit<LikedMessage, "id" | "timestamp">) => {
    const entry: LikedMessage = {
      id: crypto.randomUUID(),
      ...item,
      timestamp: new Date().toISOString(),
    };
    setLikedMessages((prev) => {
      const next = [entry, ...prev];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const removeLike = useCallback((id: string) => {
    setLikedMessages((prev) => {
      const next = prev.filter((m) => m.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  return { likedMessages, addLike, removeLike };
}
