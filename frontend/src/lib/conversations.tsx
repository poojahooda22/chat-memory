/** Client-side conversation store: the chat turns + the History list in the sidebar.
 * Persisted to localStorage so refreshing keeps your conversations. (The backend keeps the
 * durable memory; this just tracks the chat threads the UI shows.) */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { useAuth } from "./auth";

export interface Turn {
  role: "user" | "assistant";
  content: string;
  memoriesUsed?: string[];
}

export interface Conversation {
  id: string;
  title: string;
  turns: Turn[];
}

interface Store {
  conversations: Conversation[];
  activeId: string;
  active: Conversation;
  newChat: () => string;
  selectChat: (id: string) => void;
  appendTurn: (id: string, turn: Turn) => void;
  renameChat: (id: string, title: string) => void;
  deleteChat: (id: string) => void;
}

// History is per-user: the localStorage key includes the signed-in user's id, so two accounts
// in the same browser never see each other's chat threads (the durable memory is already
// per-user on the backend; this scopes the client-side thread list to match).
const storageKeyFor = (userId: string) => `chat-memory:conversations:${userId}`;
const ConversationsContext = createContext<Store | null>(null);

function newId(): string {
  return `conv-${Math.random().toString(36).slice(2)}-${Date.now()}`;
}

function loadFor(key: string): Conversation[] {
  try {
    const raw = localStorage.getItem(key);
    if (raw) {
      const parsed = JSON.parse(raw) as Conversation[];
      if (parsed.length) return parsed;
    }
  } catch {
    /* ignore */
  }
  return [{ id: newId(), title: "New chat", turns: [] }];
}

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const storageKey = storageKeyFor(user?.id ?? "anon");

  const [conversations, setConversations] = useState<Conversation[]>(() => loadFor(storageKey));
  const [activeId, setActiveId] = useState<string>(() => conversations[0]!.id);

  // when the signed-in user changes, swap to that user's own conversation list
  const currentKey = useRef(storageKey);
  useEffect(() => {
    if (currentKey.current !== storageKey) {
      currentKey.current = storageKey;
      const next = loadFor(storageKey);
      setConversations(next);
      setActiveId(next[0]!.id);
    }
  }, [storageKey]);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(conversations));
    } catch {
      /* ignore */
    }
  }, [conversations, storageKey]);

  const newChat = useCallback(() => {
    const id = newId();
    setConversations((cs) => [{ id, title: "New chat", turns: [] }, ...cs]);
    setActiveId(id);
    return id;
  }, []);

  const selectChat = useCallback((id: string) => setActiveId(id), []);

  const appendTurn = useCallback((id: string, turn: Turn) => {
    setConversations((cs) =>
      cs.map((c) => {
        if (c.id !== id) return c;
        // title the conversation from the first user message
        const title =
          c.title === "New chat" && turn.role === "user" ? turn.content.slice(0, 40) : c.title;
        return { ...c, title, turns: [...c.turns, turn] };
      }),
    );
  }, []);

  const renameChat = useCallback((id: string, title: string) => {
    setConversations((cs) => cs.map((c) => (c.id === id ? { ...c, title } : c)));
  }, []);

  const deleteChat = useCallback((id: string) => {
    setConversations((cs) => {
      const remaining = cs.filter((c) => c.id !== id);
      const next = remaining.length ? remaining : [{ id: newId(), title: "New chat", turns: [] }];
      setActiveId((current) => (current === id ? next[0]!.id : current));
      return next;
    });
  }, []);

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0]!,
    [conversations, activeId],
  );

  const value: Store = {
    conversations, activeId, active, newChat, selectChat, appendTurn, renameChat, deleteChat,
  };
  return <ConversationsContext.Provider value={value}>{children}</ConversationsContext.Provider>;
}

export function useConversations(): Store {
  const ctx = useContext(ConversationsContext);
  if (!ctx) throw new Error("useConversations must be used within a ConversationsProvider");
  return ctx;
}