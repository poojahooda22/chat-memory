import { useState } from "react";
import {
  BrainCircuit,
  Database,
  History,
  LogOut,
  MoreHorizontal,
  Orbit,
  PanelLeft,
  Pencil,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useLocation, useNavigate } from "react-router";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/lib/auth";
import { useConversations } from "@/lib/conversations";
import { cn } from "@/lib/utils";

// "Chat" is not a nav item — the New chat button and the History rows already go to the
// chat. Only Memory and Sources are separate destinations.
const NAV = [
  { to: "/memory", label: "Memory", icon: Sparkles },
  { to: "/moments", label: "Moments", icon: Orbit },
  { to: "/sources", label: "Sources", icon: Database },
];

export function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { conversations, activeId, newChat, selectChat, renameChat, deleteChat } = useConversations();
  const { user, signOut } = useAuth();
  const [collapsed, setCollapsed] = useState(false);
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

  // a conversation only appears once it has messages — an empty new chat isn't history
  const visible = conversations.filter((c) => c.turns.length > 0);

  return (
    <aside
      className={cn(
        "bg-sidebar text-sidebar-foreground flex h-full shrink-0 flex-col border-r transition-[width] duration-200",
        collapsed ? "w-[58px]" : "w-52",
      )}
    >
      {/* Brand + collapse toggle */}
      <div className={cn("flex h-14 items-center gap-2 px-3", collapsed && "justify-center px-0")}>
        {!collapsed && (
          <div className="text-foreground flex flex-1 items-center gap-2 px-1">
            <BrainCircuit className="size-5" />
            <span className="text-sm font-semibold tracking-tight">chat-memory</span>
          </div>
        )}
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="text-muted-foreground hover:bg-sidebar-accent hover:text-foreground inline-flex size-8 items-center justify-center rounded-lg transition-colors"
        >
          <PanelLeft className="size-[18px]" />
        </button>
      </div>

      {/* New chat — expanded: full-width pill; collapsed: centered circle */}
      <div className="px-3 pb-2">
        <button
          onClick={startNewChat}
          title={collapsed ? "New chat" : undefined}
          className={cn(
            "border-sidebar-border bg-sidebar-accent/40 text-sidebar-foreground flex items-center gap-3 border text-sm font-medium transition-colors",
            "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
            collapsed ? "mx-auto size-9 justify-center rounded-full" : "w-full rounded-lg px-3 py-2",
          )}
        >
          <Plus className="size-[18px] shrink-0" />
          {!collapsed && <span>New chat</span>}
        </button>
      </div>

      <nav className="mt-2 space-y-1 px-3">
        {NAV.map(({ to, label, icon: Icon }) => {
          const active = location.pathname === to;
          return (
            <button
              key={to}
              onClick={() => navigate(to)}
              title={collapsed ? label : undefined}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                collapsed && "justify-center px-0",
              )}
            >
              <Icon className="size-[18px] shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </button>
          );
        })}

        {/* Collapsed: History becomes an icon that re-expands the sidebar */}
        {collapsed && (
          <button
            onClick={() => setCollapsed(false)}
            title="History"
            className="hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground flex w-full items-center justify-center rounded-lg py-2 text-sm font-medium transition-colors"
          >
            <History className="size-[18px] shrink-0" />
          </button>
        )}
      </nav>

      {collapsed ? (
        <div className="flex-1" />
      ) : (
        <div className="mt-6 min-h-0 flex-1 overflow-y-auto px-3">
          <div className="text-muted-foreground flex items-center gap-2 px-3 py-1.5 text-xs">
            <History className="size-3.5" /> History
          </div>
          {visible.length === 0 && (
            <div className="text-muted-foreground px-3 py-1 text-xs">No conversations yet.</div>
          )}

          <ul className="space-y-0.5">
            {visible.map((c) => {
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
      )}

      <div className={cn("border-t py-3", collapsed ? "px-0" : "px-4")}>
        <div className={cn("flex items-center gap-2", collapsed && "justify-center")}>
          <div className="bg-sidebar-accent text-sidebar-accent-foreground flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold uppercase">
            {(user?.email ?? "?").charAt(0)}
          </div>
          {!collapsed && (
            <>
              <span className="flex-1 truncate text-sm font-medium" title={user?.email ?? ""}>
                {user?.email ?? "account"}
              </span>
              <button
                onClick={() => signOut()}
                title="Sign out"
                className="text-muted-foreground hover:bg-sidebar-accent hover:text-foreground rounded-md p-1 transition-colors"
              >
                <LogOut className="size-4" />
              </button>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}