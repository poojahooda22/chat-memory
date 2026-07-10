# Memory evaluation (P0)

An offline, reproducible evaluation of the memory engine on the public
[LOCOMO](https://github.com/snap-research/locomo) long-conversation QA benchmark (CC BY-NC —
**study-only**). Run from the command line; not part of the request-serving app.

## What it measures

Three systems answer the same questions; only the retrieved context differs, so a gap in the score
is a gap in *memory*, not prompt wording:

- **ours** — the memory engine: distilled semantic facts **+** the episodic layer (raw dialogue).
- **naive_rag** — chunk the conversation, embed, retrieve top-k. RAG without memory management.
- **full_context** — the entire conversation in every prompt. Quality ceiling, cost floor.

Answers are graded by an independent, stronger model (`claude-sonnet-4.6`) — the **J score** = the
fraction judged semantically correct. The judge differs from the answer model (`gpt-4o-mini`) so it
can't favor its own outputs. Three axes are reported: quality (J, overall + per category), latency
(p50/p95), and token cost.

## Headline result (2 conversations, 233 answerable questions)

| Config | J overall | tokens/question |
|---|---|---|
| **ours** | **0.506** | 1,159 |
| naive_rag | 0.455 | 1,955 |
| full_context | 0.515 | 12,952 |

- **Ours ≈ full-context quality at ~1/11th the tokens** (0.506 vs 0.515; 1,159 vs 12,952) — the
  near-full-context-quality-at-a-fraction-of-the-cost result, on this engine.
- **Ours beats naive RAG** (0.506 vs 0.455) at ~40% fewer tokens.

| Category | ours | naive_rag | full_context |
|---|---|---|---|
| temporal | **0.492** | 0.302 | 0.270 |
| single_hop | 0.623 | 0.649 | 0.719 |
| multi_hop | 0.256 | 0.209 | 0.372 |
| open_domain | 0.385 | 0.308 | 0.385 |

**Ours wins temporal outright** — dated episodic retrieval surfaces the exact turn, where full-context
must find that needle in a 13k-token haystack. Full-context wins multi-hop/single-hop (raw everything
helps connect facts across sessions).

## The finding that mattered

The first run scored ours = 0.163 — far below both baselines. Investigation (not reporting the number)
found ours abstained on 161/233 questions: the answers lived in the **episodes**, but the answer path
retrieved **only distilled facts** and never searched the episodic layer. Wiring in facts + episodes
moved ours **0.163 → 0.506 (3.1×)**. The shipped chat retrieval should search chat-episodes, not just
facts + image episodes (a product action item).

## Caveats

- 2 of LOCOMO's 10 conversations (233 of ~1,540 answerable questions) — directional, not a full-benchmark
  number. Re-run with `--samples 10` for the complete set.
- Our own setup (models, retrieval-k, prompt); not comparable to vendors' published LOCOMO numbers.
- Ours' latency is higher (two retrievals; currently embeds the query twice — a known optimization).
- Category 5 (adversarial / unanswerable) is excluded this round; it tests refusal and is scored separately.

## Run

```bash
# from backend/, with the DB up and AI_GATEWAY_API_KEY in .env
uv run python -m eval.run --samples 2                 # full: reset + ingest + measure
uv run python -m eval.run --samples 2 --skip-ingest   # re-measure over the existing store
# override the judge:  EVAL_JUDGE_MODEL=openai/gpt-4o uv run python -m eval.run ...
```

Reports + raw per-question records (question, gold, each answer, verdict) land in `results/`
(gitignored) as `locomo-<timestamp>.md` / `.json`.