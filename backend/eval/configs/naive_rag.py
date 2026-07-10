"""The 'naive_rag' baseline — retrieval without memory management.

Chop the conversation into fixed-size chunks, embed each, and at query time pull the top-k most
similar chunks into the prompt. This is "RAG" as usually meant: no fact extraction, no
deduplication, no ADD/UPDATE/DELETE — just cosine similarity over raw text. The gap between this
and 'ours' is precisely the paper's contribution: curated, conflict-resolved memory vs. raw
chunk retrieval.

Same embedding model as the product (text-embedding-3-small via the gateway), so the only
difference from 'ours' is what gets stored and retrieved, not how text is vectorized.
"""

import math
from dataclasses import dataclass

from app.config import Settings
from app.memory.embeddings import embed_text
from eval.common import AnswerResult, timed_answer
from eval.loader import Sample

CHUNK_LINES = 6  # dialogue lines per chunk — a few turns of local context each
TOP_K = 10  # chunks retrieved per question


@dataclass
class ChunkIndex:
    chunks: list[str]
    embeddings: list[list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def build_index(client, settings: Settings, sample: Sample) -> ChunkIndex:
    """Chunk the whole conversation and embed each chunk once (the RAG 'ingest')."""
    lines = sample.dialogue_lines()
    chunks = ["\n".join(lines[i : i + CHUNK_LINES]) for i in range(0, len(lines), CHUNK_LINES)]
    embeddings = [embed_text(client, settings.embedding_model, chunk) for chunk in chunks]
    return ChunkIndex(chunks=chunks, embeddings=embeddings)


def answer(
    index: ChunkIndex, client, settings: Settings, question: str
) -> AnswerResult:
    def retrieve_context() -> str:
        query_vec = embed_text(client, settings.embedding_model, question)
        ranked = sorted(
            zip(index.chunks, index.embeddings),
            key=lambda pair: _cosine(query_vec, pair[1]),
            reverse=True,
        )
        top = [chunk for chunk, _ in ranked[:TOP_K]]
        return "\n---\n".join(top) or "(no chunks)"

    return timed_answer(retrieve_context, client, settings.llm_model, question)