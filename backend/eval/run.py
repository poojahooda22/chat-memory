"""Orchestrator: run the LOCOMO eval end-to-end and write the results table.

Usage (from backend/, with the DB up and AI_GATEWAY_API_KEY in .env):
    uv run python -m eval.run --samples 2 --max-questions 40

For each conversation it ingests the dialogue into each config's store, answers every question
under all three configs (timed + token-counted), judges each answer against the gold with a
different model, then aggregates to J score / latency p50-p95 / mean tokens and writes a markdown
report plus the raw per-question records.

Study-only: LOCOMO is CC BY-NC. These are our own measurements, not comparable to vendors'
differently-configured numbers.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session

from app.config import get_settings
from app.db import build_engine
from app.llm import build_llm_client
from eval.common import reset_user
from eval.configs import full_context, naive_rag, ours
from eval.judge import judge, judge_model
from eval.loader import load_samples
from eval.metrics import Record, Summary, summarize

RESULTS_DIR = Path(__file__).parent / "results"
ALL_CONFIGS = ("ours", "naive_rag", "full_context")


def _smoke_test_judge(client, model: str) -> None:
    """Fail fast if the judge model doesn't resolve on the gateway — before a costly full run."""
    verdict = judge(client, model, "What is 2+2?", "4", "four")
    print(f"judge model '{model}' resolves (sanity: 'four' vs '4' -> correct={verdict.correct})")


def _answer_one(cfg, question, *, session, client, settings, uid, rag_index, dialogue):
    """Route one question to one config's answerer. Returns an AnswerResult."""
    if cfg == "ours":
        return ours.answer(session, client, settings, question, user_id=uid)
    if cfg == "naive_rag":
        return naive_rag.answer(rag_index, client, settings, question)
    return full_context.answer(client, settings, question, dialogue=dialogue)


def run(samples: int, max_questions: int | None, configs: tuple[str, ...], out_path: Path,
        skip_ingest: bool = False) -> None:
    settings = get_settings()
    engine = build_engine(settings)
    client = build_llm_client(settings)
    jmodel = judge_model()
    _smoke_test_judge(client, jmodel)

    data = load_samples(limit=samples)
    records: list[Record] = []

    with Session(engine) as session:
        for sample in data:
            questions = sample.qa[:max_questions] if max_questions else sample.qa
            print(f"\n=== {sample.sample_id}: {len(sample.sessions)} sessions, "
                  f"{len(questions)} questions ===")

            uid = f"eval-{sample.sample_id}"
            rag_index = None
            dialogue = sample.dialogue_lines()

            if "ours" in configs and not skip_ingest:
                print("  ingesting into our memory engine...")
                reset_user(session, uid)
                ours.ingest(session, client, settings, sample,
                            user_id=uid, conversation_id=sample.sample_id)
            elif "ours" in configs:
                print("  reusing the already-ingested store (--skip-ingest)")
            if "naive_rag" in configs:
                print("  building the RAG chunk index...")
                rag_index = naive_rag.build_index(client, settings, sample)

            print(f"  answering + judging ({len(configs)} configs)...")
            for i, qa in enumerate(questions, 1):
                for cfg in configs:
                    try:
                        result = _answer_one(
                            cfg, qa.question,
                            session=session, client=client, settings=settings,
                            uid=uid, rag_index=rag_index, dialogue=dialogue,
                        )
                        verdict = judge(client, jmodel, qa.question, qa.answer, result.text)
                        records.append(Record(
                            config=cfg, sample_id=sample.sample_id, category=qa.category_name,
                            question=qa.question, gold=qa.answer, predicted=result.text,
                            correct=verdict.correct, search_ms=result.search_ms,
                            total_ms=result.total_ms, prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                        ))
                    except Exception as exc:  # one bad call must not sink a long run
                        print(f"    ! {cfg} q{i} error: {exc}")
                        records.append(Record(
                            config=cfg, sample_id=sample.sample_id, category=qa.category_name,
                            question=qa.question, gold=qa.answer, predicted=f"<error: {exc}>",
                            correct=False, search_ms=0.0, total_ms=0.0,
                            prompt_tokens=0, completion_tokens=0,
                        ))
                if i % 10 == 0:
                    print(f"    ...{i}/{len(questions)}")

    summaries = [summarize(cfg, [r for r in records if r.config == cfg]) for cfg in configs]
    _write_report(out_path, summaries, records, jmodel)
    _print_summary(summaries)


def _write_report(path: Path, summaries: list[Summary], records: list[Record], jmodel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = sorted({r.category for r in records})
    n = summaries[0].n if summaries else 0

    lines = [
        f"# LOCOMO evaluation — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Answer model: `openai/gpt-4o-mini` · Judge model: `{jmodel}` · Questions/config: {n}",
        "",
        "Study-only — LOCOMO is CC BY-NC. These are our own measurements under one setup; they are "
        "not comparable to vendors' differently-configured published numbers.",
        "",
        "## Quality — LLM-as-judge J score (1 = judged correct)",
        "",
        "| Config | N | J overall | " + " | ".join(categories) + " |",
        "|" + "---|" * (3 + len(categories)),
    ]
    for s in summaries:
        cells = " | ".join(f"{s.j_by_category.get(c, 0.0):.3f}" for c in categories)
        lines.append(f"| {s.config} | {s.n} | **{s.j_overall:.3f}** | {cells} |")

    lines += [
        "",
        "## Cost & latency (per question)",
        "",
        "| Config | search p50 (ms) | search p95 (ms) | total p50 (ms) | total p95 (ms) | mean tokens |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.config} | {s.search_p50:.0f} | {s.search_p95:.0f} | "
            f"{s.total_p50:.0f} | {s.total_p95:.0f} | {s.mean_total_tokens:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw_path = path.with_suffix(".json")
    raw_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    print(f"\nreport  -> {path}\nrecords -> {raw_path}")


def _print_summary(summaries: list[Summary]) -> None:
    print("\n" + "=" * 60)
    for s in summaries:
        print(f"{s.config:>13}  J={s.j_overall:.3f}  total_p50={s.total_p50:.0f}ms  "
              f"tokens/q={s.mean_total_tokens:.0f}  (n={s.n})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LOCOMO memory eval.")
    parser.add_argument("--samples", type=int, default=2, help="how many of the 10 conversations")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="cap questions per conversation (0 = all)")
    parser.add_argument("--configs", type=str, default=",".join(ALL_CONFIGS),
                        help="comma-separated subset of: " + ",".join(ALL_CONFIGS))
    parser.add_argument("--out", type=str, default="", help="output .md path (default: timestamped)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="reuse the already-ingested store instead of resetting + re-ingesting")
    args = parser.parse_args()

    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip() in ALL_CONFIGS)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"locomo-{stamp}.md"

    run(
        samples=args.samples,
        max_questions=args.max_questions or None,
        configs=configs,
        out_path=out_path,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()