import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";

import { getQuiz, submitQuiz } from "@/lib/api";

/** Cold-start onboarding: a few skippable questions whose answers seed memory at LOW confidence
 * (a gentle guess) — so anything you later say in chat supersedes them. */
export function Quiz() {
  const queryClient = useQueryClient();
  const questions = useQuery({ queryKey: ["quiz"], queryFn: getQuiz });
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const submit = useMutation({
    mutationFn: (payload: { question: string; answer: string }[]) => submitQuiz(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memories"] }),
  });

  function onSubmit() {
    const payload = (questions.data ?? [])
      .map((q) => ({ question: q.prompt, answer: (answers[q.id] ?? "").trim() }))
      .filter((a) => a.answer);
    if (payload.length) submit.mutate(payload);
  }

  return (
    <div className="min-h-0 w-full flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-2xl px-4 py-8">
        <h2 className="text-lg font-semibold">About you</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Answer a few to give your memory a head start. Skip anything — these are treated as gentle
          guesses (low confidence), so what you actually say in chat always wins.
        </p>

        {submit.isSuccess ? (
          <div className="bg-card mt-6 rounded-2xl border p-6 text-center">
            <Sparkles className="text-muted-foreground mx-auto size-6" />
            <p className="mt-2 text-sm font-medium">Thanks — building your memory…</p>
            <p className="text-muted-foreground mt-1 text-xs">
              Your answers are becoming memories. Check the Memory page in a moment.
            </p>
          </div>
        ) : (
          <div className="mt-6 space-y-4">
            {(questions.data ?? []).map((q) => (
              <div key={q.id} className="bg-card rounded-2xl border p-4">
                <label className="text-sm font-medium">{q.prompt}</label>
                <textarea
                  value={answers[q.id] ?? ""}
                  onChange={(e) => setAnswers((a) => ({ ...a, [q.id]: e.target.value }))}
                  rows={2}
                  placeholder="Your answer (optional)…"
                  className="border-input bg-background focus:border-ring/60 mt-2 w-full resize-none rounded-lg border px-3 py-2 text-sm focus:outline-none"
                />
              </div>
            ))}
            <button
              onClick={onSubmit}
              disabled={submit.isPending}
              className="bg-primary text-primary-foreground rounded-lg px-4 py-2 text-sm font-medium transition-opacity hover:opacity-90 disabled:opacity-60"
            >
              {submit.isPending ? "Saving…" : "Save to memory"}
            </button>
            {submit.isError && (
              <p className="text-destructive text-xs">Failed: {(submit.error as Error).message}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}