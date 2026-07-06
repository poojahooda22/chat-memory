import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { sendChat, type ChatResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Turn {
  role: "user" | "assistant";
  content: string;
  memoriesUsed?: string[];
}

export function ChatPanel({ conversationId }: { conversationId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const queryClient = useQueryClient();
  const scrollRef = useRef<HTMLDivElement>(null);

  const mutation = useMutation({
    mutationFn: (message: string) => sendChat(message, conversationId),
    onSuccess: (res: ChatResponse) => {
      setTurns((t) => [
        ...t,
        { role: "assistant", content: res.reply, memoriesUsed: res.memories_used },
      ]);
      // the exchange may have changed memory — refresh the bubbles
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      requestAnimationFrame(() =>
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }),
      );
    },
  });

  function send() {
    const message = draft.trim();
    if (!message || mutation.isPending) return;
    setTurns((t) => [...t, { role: "user", content: message }]);
    setDraft("");
    mutation.mutate(message);
    requestAnimationFrame(() =>
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }),
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {turns.length === 0 && (
          <div className="text-muted-foreground mt-16 text-center text-sm">
            <Sparkles className="mx-auto mb-3 size-6 opacity-60" />
            Tell me about yourself — I&apos;ll remember it, and you can watch the memory grow on the right.
          </div>
        )}
        {turns.map((turn, i) => (
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
        {mutation.isPending && (
          <div className="text-muted-foreground text-xs">thinking…</div>
        )}
      </div>

      <div className="border-t p-3">
        <div className="relative">
          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Say something about yourself, or ask what I remember…"
            className="max-h-40 resize-none pr-12"
            rows={2}
          />
          <Button
            size="icon-sm"
            onClick={send}
            disabled={!draft.trim() || mutation.isPending}
            className="absolute right-2 bottom-2"
          >
            <ArrowUp />
          </Button>
        </div>
      </div>
    </div>
  );
}