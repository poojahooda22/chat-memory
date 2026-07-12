import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Paperclip } from "lucide-react";
import { useNavigate } from "react-router";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { streamChat, uploadImages } from "@/lib/api";
import { useConversations } from "@/lib/conversations";
import { cn } from "@/lib/utils";

// markdown body styling — shared by rendered turns and the live streaming bubble
const MD_BODY =
  "break-words [&_p]:my-1 [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_a]:underline [&_code]:font-mono [&_code]:text-[12.5px] [&_code]:break-words [&_pre]:my-2 [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-background/60 [&_pre]:p-3 [&_pre_code]:break-normal";

export function Home() {
  const { active, appendTurn } = useConversations();
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState<string | null>(null); // null = idle; else partial reply
  const [uploadNote, setUploadNote] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const empty = active.turns.length === 0;

  const imageUpload = useMutation({
    mutationFn: uploadImages,
    onSuccess: (jobs) => {
      queryClient.invalidateQueries({ queryKey: ["uploads"] });
      setUploadNote(`${jobs.length} image${jobs.length > 1 ? "s" : ""} feeding your memory`);
    },
    onError: (err: Error) => setUploadNote(`upload failed: ${err.message}`),
  });

  function scrollToBottom() {
    requestAnimationFrame(() =>
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }),
    );
  }

  async function send(message: string) {
    const text = message.trim();
    if (!text || streaming !== null) return;
    const convId = active.id;
    appendTurn(convId, { role: "user", content: text });
    setDraft("");
    setStreaming(""); // "" = waiting for the first token (renders "thinking…")
    scrollToBottom();

    let full = "";
    try {
      await streamChat(text, convId, (chunk) => {
        full += chunk;
        setStreaming(full);
        scrollToBottom();
      });
      appendTurn(convId, { role: "assistant", content: full || "(no response)" });
    } catch {
      appendTurn(convId, {
        role: "assistant",
        content: "⚠️ Something went wrong — please try again.",
      });
    } finally {
      setStreaming(null);
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      scrollToBottom();
    }
  }

  // Perplexity-style composer: an elevated bg-card shell with an auto-growing bare textarea
  const composer = (
    <div className="w-full">
      {uploadNote && (
        <button
          type="button"
          onClick={() => {
            setUploadNote(null);
            navigate("/sources");
          }}
          className="text-muted-foreground hover:text-foreground mb-1.5 text-xs underline-offset-2 hover:underline"
        >
          {uploadNote} — view in Sources
        </button>
      )}
    <form
      onSubmit={(e) => {
        e.preventDefault();
        send(draft);
      }}
      className="border-border bg-card focus-within:border-ring/60 w-full rounded-2xl border px-3 py-2"
    >
      <div className="flex items-end gap-2">
        <button
          type="button"
          aria-label="Add photos to memory"
          onClick={() => fileRef.current?.click()}
          className="text-muted-foreground hover:bg-secondary hover:text-foreground mb-0.5 inline-flex size-8 shrink-0 items-center justify-center rounded-full transition-colors"
        >
          <Paperclip className="size-4" />
        </button>
        {/* raw File objects go straight to FormData — EXIF survives (no canvas re-encode) */}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []).filter((f) =>
              f.type.startsWith("image/"),
            );
            if (files.length) imageUpload.mutate(files);
            e.target.value = "";
          }}
        />
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(draft);
            }
          }}
          rows={1}
          placeholder="Ask anything…"
          className="field-sizing-content text-foreground placeholder:text-muted-foreground block max-h-[30vh] min-h-[24px] flex-1 resize-none overflow-y-auto bg-transparent py-1.5 text-[15px] focus:outline-none"
        />
        <button
          type="submit"
          aria-label="Send"
          disabled={!draft.trim() || streaming !== null}
          className={cn(
            "inline-flex size-8 shrink-0 items-center justify-center rounded-full transition-colors",
            draft.trim() && streaming === null
              ? "bg-primary text-primary-foreground hover:opacity-90"
              : "bg-secondary text-muted-foreground",
          )}
        >
          <ArrowUp className="size-4" />
        </button>
      </div>
    </form>
    </div>
  );

  // ── Empty state: centered composer (Perplexity/Lumina home) ──
  if (empty) {
    return (
      <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-6 px-4">
        <h1 className="text-3xl font-semibold tracking-tight">chat-memory</h1>
        <p className="text-muted-foreground -mt-3 text-sm">
          an assistant that remembers you — and shows its receipts
        </p>
        {composer}
      </div>
    );
  }

  // ── Conversation view: the scroll container is FULL-WIDTH (scrollbar at the page edge,
  //    matching the sibling app); the centered column lives inside it; composer pinned below ──
  return (
    <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-6">
        {active.turns.map((turn, i) => (
          <div key={i} className={cn("flex", turn.role === "user" ? "justify-end" : "justify-start")}>
            <div
              className={cn(
                "max-w-[85%] min-w-0 overflow-hidden rounded-xl px-4 py-2.5 text-sm",
                turn.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground",
              )}
            >
              {/* markdown body — long lines wrap, code blocks scroll inside the bubble */}
              <div className={MD_BODY}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
          {streaming !== null &&
            (streaming === "" ? (
              <div className="text-muted-foreground text-xs">thinking…</div>
            ) : (
              <div className="flex justify-start">
                <div className="bg-secondary text-secondary-foreground max-w-[85%] min-w-0 overflow-hidden rounded-xl px-4 py-2.5 text-sm">
                  <div className={MD_BODY}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{streaming}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>
      <div className="shrink-0 px-4 py-3">
        <div className="mx-auto w-full max-w-3xl">{composer}</div>
      </div>
    </div>
  );
}