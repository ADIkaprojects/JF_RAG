import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import type { ChatMessage } from "@/components/ChatView";

export interface Session {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

const SESSIONS_INDEX_KEY = "rag_sessions_index";
const SESSION_KEY_PREFIX = "rag_session_";
const MAX_SESSIONS = 50;

const isBrowser = typeof window !== "undefined";

function reviveMessage(raw: any): ChatMessage {
  return {
    ...raw,
    timestamp: raw.timestamp ? new Date(raw.timestamp) : new Date(),
  };
}

function safeParseSession(raw: string): Session | null {
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || !parsed.id) return null;

    const messages = Array.isArray(parsed.messages)
      ? (parsed.messages as any[]).map(reviveMessage)
      : [];

    return {
      id: String(parsed.id),
      title: typeof parsed.title === "string" ? parsed.title : "New Conversation",
      createdAt:
        typeof parsed.createdAt === "string"
          ? parsed.createdAt
          : new Date().toISOString(),
      updatedAt:
        typeof parsed.updatedAt === "string"
          ? parsed.updatedAt
          : new Date().toISOString(),
      messages,
    };
  } catch {
    return null;
  }
}

function loadSessionsFromStorage(): Session[] {
  if (!isBrowser) return [];

  try {
    const indexRaw = window.localStorage.getItem(SESSIONS_INDEX_KEY);
    const ids = indexRaw ? (JSON.parse(indexRaw) as string[]) : [];
    if (!Array.isArray(ids)) return [];

    const sessions: Session[] = [];

    for (const id of ids) {
      const raw = window.localStorage.getItem(SESSION_KEY_PREFIX + id);
      if (!raw) continue;
      const session = safeParseSession(raw);
      if (session) {
        sessions.push(session);
      }
    }

    return sessions;
  } catch {
    return [];
  }
}

function deriveTitle(messages: ChatMessage[]): string | null {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return null;
  const trimmed = firstUser.content.trim();
  if (!trimmed) return null;
  if (trimmed.length <= 55) return trimmed;
  return trimmed.slice(0, 55) + "…";
}

export function useConversationHistory() {
  const [sessions, setSessions] = useState<Session[]>([]);

  useEffect(() => {
    if (!isBrowser) return;
    setSessions(loadSessionsFromStorage());
  }, []);

  const persistIndex = useCallback((ids: string[]) => {
    if (!isBrowser) return;
    try {
      window.localStorage.setItem(SESSIONS_INDEX_KEY, JSON.stringify(ids));
    } catch {
      // ignore
    }
  }, []);

  const saveSession = useCallback(
    (session: Session) => {
      if (!isBrowser || !session.id) return;

      // If there are no messages, do not persist an "empty" conversation.
      // Instead, remove any existing stored session for this id.
      if (!session.messages || session.messages.length === 0) {
        try {
          window.localStorage.removeItem(SESSION_KEY_PREFIX + session.id);

          const indexRaw = window.localStorage.getItem(SESSIONS_INDEX_KEY);
          let ids = indexRaw ? (JSON.parse(indexRaw) as string[]) : [];
          if (!Array.isArray(ids)) ids = [];

          ids = ids.filter((existingId) => existingId !== session.id);
          persistIndex(ids);

          setSessions(loadSessionsFromStorage());
        } catch {
          // ignore
        }
        return;
      }

      const nowIso = new Date().toISOString();

      const titleFromMessages = deriveTitle(session.messages);
      const title =
        session.title && session.title.trim().length > 0
          ? session.title
          : titleFromMessages || "New Conversation";

      const createdAt =
        session.createdAt && session.createdAt.length > 0
          ? session.createdAt
          : nowIso;

      const updated: Session = {
        ...session,
        title,
        createdAt,
        updatedAt: nowIso,
      };

      try {
        window.localStorage.setItem(
          SESSION_KEY_PREFIX + updated.id,
          JSON.stringify(updated)
        );

        const indexRaw = window.localStorage.getItem(SESSIONS_INDEX_KEY);
        let ids = indexRaw ? (JSON.parse(indexRaw) as string[]) : [];
        if (!Array.isArray(ids)) ids = [];

        // Move id to front (newest first)
        ids = [updated.id, ...ids.filter((id) => id !== updated.id)];

        if (ids.length > MAX_SESSIONS) {
          const toRemove = ids.slice(MAX_SESSIONS);
          ids = ids.slice(0, MAX_SESSIONS);
          for (const removeId of toRemove) {
            window.localStorage.removeItem(SESSION_KEY_PREFIX + removeId);
          }
        }

        persistIndex(ids);
        setSessions(loadSessionsFromStorage());
      } catch (err) {
        const isQuota =
          err instanceof DOMException &&
          (err.name === "QuotaExceededError" || err.name === "NS_ERROR_DOM_QUOTA_REACHED");
        if (isQuota) {
          toast.error("Storage full", {
            description: "Conversation history could not be saved — browser storage is full. Try clearing old conversations.",
          });
        }
      }
    },
    [persistIndex]
  );

  const deleteSession = useCallback(
    (id: string) => {
      if (!isBrowser) return;

      try {
        window.localStorage.removeItem(SESSION_KEY_PREFIX + id);

        const indexRaw = window.localStorage.getItem(SESSIONS_INDEX_KEY);
        let ids = indexRaw ? (JSON.parse(indexRaw) as string[]) : [];
        if (!Array.isArray(ids)) ids = [];

        ids = ids.filter((existingId) => existingId !== id);
        persistIndex(ids);

        setSessions(loadSessionsFromStorage());
      } catch {
        // ignore
      }
    },
    [persistIndex]
  );

  const loadSession = useCallback((id: string): Session | null => {
    if (!isBrowser) return null;

    const raw = window.localStorage.getItem(SESSION_KEY_PREFIX + id);
    if (!raw) return null;
    return safeParseSession(raw);
  }, []);

  const createNewSession = useCallback((): string => {
    const id = crypto.randomUUID();
    // We purposefully do NOT persist empty sessions here.
    // The session is only written once it has at least one message in saveSession().
    return id;
  }, []);

  return {
    sessions,
    saveSession,
    deleteSession,
    loadSession,
    createNewSession,
  };
}

