"""The three systems under comparison: ours (Mem0 engine), naive_rag, full_context.

Each module exposes the same two functions — ingest(sample) and answer(question) — so the
orchestrator drives them identically and only the memory strategy varies.
"""