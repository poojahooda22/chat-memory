"""Shared pieces for the three eval configs.

All three configs (ours / naive_rag / full_context) use the SAME answer prompt and the SAME
timed generation call — they differ ONLY in what context they put in front of the question.
Holding the answer prompt constant is what makes the comparison fair: it isolates the one
variable under test (the quality of the retrieved context), so a gap in the J score reflects a
gap in memory/retrieval, not in prompt wording.
"""

from dataclasses import dataclass
from time import perf_counter

from sqlmodel import Session, col, delete, select

from app.models import Episode, Memory, MemoryHistory


@dataclass
class AnswerResult:
    text: str
    search_ms: float  # time spent building the context (retrieval)
    total_ms: float  # retrieve + generate — the paper's "total latency"
    prompt_tokens: int
    completion_tokens: int


# Config-neutral answer prompt. Deliberately terse, because LOCOMO gold answers are short
# ("7 May 2023", "Psychology") — a chatty answer would score the same to a judge but inflate
# token cost and blur the comparison.
ANSWER_SYSTEM = (
    "You answer a question about a person using ONLY the context provided. "
    "Answer in as few words as possible — a name, a date, a short phrase — matching the style of "
    "a QA benchmark's ground-truth answers. If the context does not contain the answer, reply "
    'exactly "I don\'t know." Do not explain and do not add a sentence.'
)


def build_answer_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ]


def timed_answer(retrieve_context, client, model: str, question: str) -> AnswerResult:
    """Run a config's retrieve-then-generate under one timer.

    retrieve_context is a zero-arg closure each config supplies — it returns the context string
    (our memories, RAG chunks, or the whole conversation). We time the retrieval separately
    (search latency) and the whole thing (total latency), and read token cost off the API's own
    usage report (exact, no estimation).
    """
    t0 = perf_counter()
    context = retrieve_context()
    search_ms = (perf_counter() - t0) * 1000

    resp = client.chat.completions.create(
        model=model, messages=build_answer_messages(context, question), temperature=0
    )
    total_ms = (perf_counter() - t0) * 1000

    usage = resp.usage
    text = (resp.choices[0].message.content or "").strip()
    return AnswerResult(
        text=text,
        search_ms=search_ms,
        total_ms=total_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def reset_user(session: Session, user_id: str) -> None:
    """Delete a benchmark user's memories (+ their audit rows) and episodes so a re-run starts
    from a clean store.

    memory_history has a foreign key to memories, so its rows MUST be deleted before the memories
    they reference — otherwise Postgres rejects the parent delete. Only these tables need clearing:
    chat ingest never creates entities/relationships (those come from image labeling), and no
    ConversationSummary is written during eval ingest (the summary refresher is a route-side
    background task, not part of record_exchange).
    """
    memory_ids = session.exec(select(Memory.id).where(Memory.user_id == user_id)).all()
    if memory_ids:
        session.exec(delete(MemoryHistory).where(col(MemoryHistory.memory_id).in_(memory_ids)))
    session.exec(delete(Memory).where(col(Memory.user_id) == user_id))
    session.exec(delete(Episode).where(col(Episode.user_id) == user_id))
    session.commit()