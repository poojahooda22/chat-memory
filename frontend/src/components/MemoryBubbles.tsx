import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { History, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { deleteMemory, getMemoryHistory, listMemories, type Memory } from "@/lib/api";

/** The live fact-sheet as floating "bubbles" — click one to see its audit trail. */
export function MemoryBubbles() {
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ["memories"],
    queryFn: listMemories,
  });
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-5 py-3">
        <h2 className="text-sm font-semibold">what I remember about you</h2>
        <span className="text-muted-foreground text-xs">{memories.length} memories</span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-5">
        {isLoading && <div className="text-muted-foreground text-sm">loading…</div>}
        {!isLoading && memories.length === 0 && (
          <div className="text-muted-foreground mt-16 text-center text-sm">
            No memories yet. Chat on the left and watch them appear here.
          </div>
        )}

        <div className="flex flex-wrap gap-3">
          <AnimatePresence mode="popLayout">
            {memories.map((memory) => (
              <MemoryBubble
                key={memory.id}
                memory={memory}
                open={openId === memory.id}
                onToggle={() => setOpenId((id) => (id === memory.id ? null : memory.id))}
              />
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

function MemoryBubble({
  memory,
  open,
  onToggle,
}: {
  memory: Memory;
  open: boolean;
  onToggle: () => void;
}) {
  const queryClient = useQueryClient();

  const history = useQuery({
    queryKey: ["history", memory.id],
    queryFn: () => getMemoryHistory(memory.id),
    enabled: open, // only fetch the audit trail when the bubble is opened
  });

  const remove = useMutation({
    mutationFn: () => deleteMemory(memory.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memories"] }),
  });

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.85 }}
      transition={{ type: "spring", stiffness: 320, damping: 26 }}
      className={open ? "w-full" : ""}
    >
      <button
        onClick={onToggle}
        className="bg-card hover:border-ring/60 flex items-center gap-2 rounded-full border px-4 py-2 text-left text-sm shadow-sm transition-colors"
      >
        {memory.content}
        <History className="text-muted-foreground size-3.5" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-card mt-2 overflow-hidden rounded-xl border"
          >
            <div className="p-4">
              <div className="text-muted-foreground mb-2 text-xs font-medium">
                audit trail (the receipts)
              </div>
              {history.isLoading && <div className="text-muted-foreground text-xs">loading…</div>}
              <ol className="space-y-1.5">
                {history.data?.map((entry, i) => (
                  <li key={i} className="text-xs">
                    <span className="font-mono font-medium">{entry.event}</span>
                    {entry.old_content && (
                      <span className="text-muted-foreground"> · {entry.old_content} →</span>
                    )}{" "}
                    {entry.new_content && <span>{entry.new_content}</span>}
                  </li>
                ))}
              </ol>
              <div className="text-muted-foreground mt-3 text-[11px]">
                from {memory.source_episode_ids.length} source episode(s)
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => remove.mutate()}
                disabled={remove.isPending}
                className="text-destructive mt-2 -ml-2"
              >
                <Trash2 /> forget this
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}