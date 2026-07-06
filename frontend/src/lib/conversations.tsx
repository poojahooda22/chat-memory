/** Client-side conversation store: the chat turns + the History list in the sidebar.
 * Persisted to localStorage so refreshing keeps your conversations. (The backend keeps the
 * durable memory; this just tracks the chat threads the UI shows.) */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

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
}

const STORAGE_KEY = "chat-memory:conversations";
const ConversationsContext = createContext<Store | null>(null);

function newId(): string {
  return `conv-${Math.random().toString(36).slice(2)}-${Date.now()}`;
}

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as Conversation[];
  } catch {
    /* ignore */
  }
  return [];
}

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const loaded = load();
    return loaded.length ? loaded : [{ id: newId(), title: "New chat", turns: [] }];
  });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]!.id);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch {
      /* ignore */
    }
  }, [conversations]);

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

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0]!,
    [conversations, activeId],
  );

  const value: Store = { conversations, activeId, active, newChat, selectChat, appendTurn };
  return <ConversationsContext.Provider value={value}>{children}</ConversationsContext.Provider>;
}

export function useConversations(): Store {
  const ctx = useContext(ConversationsContext);
  if (!ctx) throw new Error("useConversations must be used within a ConversationsProvider");
  return ctx;
}