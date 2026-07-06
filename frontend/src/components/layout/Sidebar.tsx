import { BrainCircuit, Database, History, MessageSquare, Plus, Sparkles } from "lucide-react";
import { useLocation, useNavigate } from "react-router";

import { useConversations } from "@/lib/conversations";
import { USER_ID } from "@/lib/config";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Chat", icon: MessageSquare },
  { to: "/memory", label: "Memory", icon: Sparkles },
  { to: "/sources", label: "Sources", icon: Database },
];

export function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { conversations, activeId, newChat, selectChat } = useConversations();

  function startNewChat() {
    newChat();
    navigate("/");
  }

  return (
    <aside className="bg-sidebar text-sidebar-foreground flex h-full w-64 shrink-0 flex-col border-r">
      <div className="flex items-center gap-2 px-4 py-4">
        <BrainCircuit className="text-foreground size-5" />
        <span className="text-foreground text-sm font-semibold tracking-tight">chat-memory</span>
      </div>

      <div className="px-3">
        <button
          onClick={startNewChat}
          className="bg-sidebar-accent text-sidebar-accent-foreground hover:opacity-90 flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-opacity"
        >
          <Plus className="size-4" /> New chat
        </button>
      </div>

      <nav className="mt-4 space-y-1 px-3">
        {NAV.map(({ to, label, icon: Icon }) => {
          const active = location.pathname === to;
          return (
            <button
              key={to}
              onClick={() => navigate(to)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
              )}
            >
              <Icon className="size-[18px]" /> {label}
            </button>
          );
        })}
      </nav>

      <div className="mt-6 min-h-0 flex-1 overflow-y-auto px-3">
        <div className="text-muted-foreground flex items-center gap-2 px-3 py-1.5 text-xs">
          <History className="size-3.5" /> History
        </div>
        {conversations.length === 0 && (
          <div className="text-muted-foreground px-3 py-1 text-xs">No conversations yet.</div>
        )}
        {conversations.map((c) => (
          <button
            key={c.id}
            onClick={() => {
              selectChat(c.id);
              navigate("/");
            }}
            className={cn(
              "block w-full truncate rounded-lg px-3 py-1.5 text-left text-sm transition-colors",
              c.id === activeId && location.pathname === "/"
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "hover:bg-sidebar-accent/60",
            )}
          >
            {c.title}
          </button>
        ))}
      </div>

      <div className="border-t px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="bg-sidebar-accent text-sidebar-accent-foreground flex size-7 items-center justify-center rounded-full text-xs font-semibold uppercase">
            {USER_ID.charAt(0)}
          </div>
          <span className="text-sm font-medium capitalize">{USER_ID}</span>
        </div>
      </div>
    </aside>
  );
}