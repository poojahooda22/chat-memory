import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { sendChat, type ChatResponse } from "@/lib/api";
import { useConversations } from "@/lib/conversations";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "Hi! I'm Pooja, a backend developer who codes in Go.",
  "What do you remember about me?",
  "I love cycling on weekends.",
];

export function Home() {
  const { active, appendTurn } = useConversations();
  const [draft, setDraft] = useState("");
  const queryClient = useQueryClient();
  const scrollRef = useRef<HTMLDivElement>(null);
  const empty = active.turns.length === 0;

  const mutation = useMutation({
    mutationFn: (message: string) => sendChat(message, active.id),
    onSuccess: (res: ChatResponse) => {
      appendTurn(active.id, {
        role: "assistant",
        content: res.reply,
        memoriesUsed: res.memories_used,
      });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      scrollToBottom();
    },
  });

  function scrollToBottom() {
    requestAnimationFrame(() =>
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }),
    );
  }

  function send(message: string) {
    const text = message.trim();
    if (!text || mutation.isPending) return;
    appendTurn(active.id, { role: "user", content: text });
    setDraft("");
    mutation.mutate(text);
    scrollToBottom();
  }

  const composer = (
    <div className="relative w-full">
      <Textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send(draft);
          }
        }}
        placeholder="Ask anything…"
        className="max-h-40 resize-none rounded-xl pr-12"
        rows={2}
      />
      <Button
        size="icon-sm"
        onClick={() => send(draft)}
        disabled={!draft.trim() || mutation.isPending}
        className="absolute right-2.5 bottom-2.5"
      >
        <ArrowUp />
      </Button>
    </div>
  );

  // ── Empty state: centered composer + suggestions (Perplexity/Lumina home) ──
  if (empty) {
    return (
      <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-6 px-4">
        <h1 className="text-3xl font-semibold tracking-tight">chat-memory</h1>
        <p className="text-muted-foreground -mt-3 text-sm">
          an assistant that remembers you — and shows its receipts
        </p>
        {composer}
        <div className="flex flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="bg-card hover:border-ring/60 rounded-full border px-3.5 py-1.5 text-xs transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ── Conversation view ──
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-6">
        {active.turns.map((turn, i) => (
          <div key={i} className={cn("flex", turn.role === "user" ? "justify-end" : "justify-start")}>
            <div
              className={cn(
                "max-w-[85%] rounded-xl px-4 py-2.5 text-sm",
                turn.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground",
              )}
            >
              <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
              </div>
              {turn.memoriesUsed && turn.memoriesUsed.length > 0 && (
                <div className="border-border/60 text-muted-foreground mt-2 border-t pt-1.5 text-[11px]">
                  recalled: {turn.memoriesUsed.join(" · ")}
                </div>
              )}
            </div>
          </div>
        ))}
        {mutation.isPending && <div className="text-muted-foreground text-xs">thinking…</div>}
      </div>
      <div className="px-4 pb-4">{composer}</div>
    </div>
  );
}