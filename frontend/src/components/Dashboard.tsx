import { useState } from "react";
import { BrainCircuit } from "lucide-react";

import { ChatPanel } from "@/components/ChatPanel";
import { MemoryBubbles } from "@/components/MemoryBubbles";

/** Two-panel layout: chat on the left, the memory "bubbles" canvas on the right. */
export function Dashboard() {
  // one conversation id for the whole session (a real app would let you pick/rename)
  const [conversationId] = useState(() => `web-${Date.now()}`);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-2 border-b px-5 py-3">
        <BrainCircuit className="size-5 text-primary" />
        <h1 className="text-sm font-semibold tracking-tight">chat-memory</h1>
        <span className="text-muted-foreground ml-2 text-xs">
          an assistant that remembers you — and shows its receipts
        </span>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_1.1fr]">
        <div className="min-h-0 border-r">
          <ChatPanel conversationId={conversationId} />
        </div>
        <div className="min-h-0">
          <MemoryBubbles />
        </div>
      </div>
    </div>
  );
}