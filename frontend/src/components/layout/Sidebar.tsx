import { useState } from "react";
import { BrainCircuit, Database, History, MoreHorizontal, Pencil, Plus, Sparkles, Trash2 } from "lucide-react";
import { useLocation, useNavigate } from "react-router";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useConversations } from "@/lib/conversations";
import { USER_ID } from "@/lib/config";
import { cn } from "@/lib/utils";

// "Chat" is not a nav item — the New chat button and the History rows already go to the
// chat. Only Memory and Sources are separate destinations.
const NAV = [
  { to: "/memory", label: "Memory", icon: Sparkles },
  { to: "/sources", label: "Sources", icon: Database },
];

export function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { conversations, activeId, newChat, selectChat, renameChat, deleteChat } = useConversations();
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  function startNewChat() {
    newChat();
    navigate("/");
  }
  function startRename(id: string, current: string) {
    setRenamingId(id);
    setRenameValue(current);
  }
  function commitRename() {
    if (renamingId && renameValue.trim()) renameChat(renamingId, renameValue.trim());
    setRenamingId(null);
    setRenameValue("");
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
          className="bg-sidebar-accent text-sidebar-accent-foreground flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-opacity hover:opacity-90"
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
        {/* a conversation only appears once it has messages — an empty new chat isn't history */}
        {conversations.every((c) => c.turns.length === 0) && (
          <div className="text-muted-foreground px-3 py-1 text-xs">No conversations yet.</div>
        )}

        <ul className="space-y-0.5">
          {conversations.filter((c) => c.turns.length > 0).map((c) => {
            const isActive = c.id === activeId && location.pathname === "/";

            if (renamingId === c.id) {
              return (
                <li key={c.id}>
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onFocus={(e) => e.currentTarget.select()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") { e.preventDefault(); commitRename(); }
                      else if (e.key === "Escape") { e.preventDefault(); setRenamingId(null); }
                    }}
                    onBlur={commitRename}
                    className="border-ring/60 bg-background text-foreground w-full rounded-lg border px-3 py-1.5 text-sm focus:outline-none"
                  />
                </li>
              );
            }

            return (
              <li key={c.id} className="group/item relative">
                <button
                  onClick={() => {
                    selectChat(c.id);
                    navigate("/");
                  }}
                  title={c.title}
                  className={cn(
                    "block w-full truncate rounded-lg py-1.5 pr-8 pl-3 text-left text-sm transition-colors",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "hover:bg-sidebar-accent/60",
                  )}
                >
                  {c.title}
                </button>

                {/* three-dot menu — appears on hover (or when its own menu is open) */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      aria-label="Conversation options"
                      className="text-muted-foreground hover:bg-sidebar-accent hover:text-foreground data-[state=open]:bg-sidebar-accent absolute top-1/2 right-1 flex size-6 -translate-y-1/2 items-center justify-center rounded-md opacity-0 transition-opacity group-hover/item:opacity-100 data-[state=open]:opacity-100"
                    >
                      <MoreHorizontal className="size-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onSelect={() => startRename(c.id, c.title)}>
                      <Pencil /> Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem variant="destructive" onSelect={() => deleteChat(c.id)}>
                      <Trash2 /> Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </li>
            );
          })}
        </ul>
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