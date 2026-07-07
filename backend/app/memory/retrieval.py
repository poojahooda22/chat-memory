"""Hybrid retrieval — the OmniQuery-style read path.

Plain cosine similarity answers "what do you know about my dog", but silently fails a
date / person / place question: a year isn't a topic, so similarity can't filter by it, and a
fixed top-k drops photos an aggregate ("how many") needs. So we DECOMPOSE the question into
structured filters (entity, time range, place), FILTER the store exactly (a real WHERE — all
matches, not a top-k), then RANK what's left by similarity. A general question with no filters
falls back to plain cosine so we never over-filter.

(OmniQuery also infers multi-photo "events" and a habit layer — those are a later move. This is
the decompose → filter → rank core, which is the actual correctness fix.)
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlmodel import Session, col, func, select

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.pipeline import search_image_episodes, search_memories
from app.memory.prompts import build_decompose_messages, parse_json
from app.models import Entity, Episode, EpisodeEntity, Memory

FILTERED_CAP = 50   # an aggregate ("all" / "how many") needs the full filtered set, not a top-k
RANKED_TOP_K = 8    # for a semantic question inside a filter, how many to rank in
FACT_TOP_K = 5      # distilled facts injected alongside, as today


@dataclass
class QuerySpec:
    entities: list[str] = field(default_factory=list)
    time_range: tuple[date, date] | None = None
    place: str | None = None
    semantic_query: str = ""
    wants_all: bool = False  # a count / "all" / "every" question → return the whole filtered set

    @property
    def has_filters(self) -> bool:
        return bool(self.entities) or self.time_range is not None or self.place is not None


@dataclass
class RetrievalResult:
    facts: list[Memory]
    photos: list[Episode]
    spec: QuerySpec


def _chat_json(client, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return parse_json(response.choices[0].message.content)


def decompose_query(client, settings: Settings, message: str) -> QuerySpec:
    """One LLM call → the structured filters a question implies (entity / time / place)."""
    today = datetime.now(UTC).date().isoformat()
    raw = _chat_json(client, settings.llm_model, build_decompose_messages(message, today))

    time_range: tuple[date, date] | None = None
    tr = raw.get("time_range")
    if isinstance(tr, dict) and tr.get("start") and tr.get("end"):
        try:
            time_range = (date.fromisoformat(tr["start"]), date.fromisoformat(tr["end"]))
        except (ValueError, TypeError):
            time_range = None

    entities = [e.strip() for e in (raw.get("entities") or []) if isinstance(e, str) and e.strip()]
    place = raw.get("place")
    return QuerySpec(
        entities=entities,
        time_range=time_range,
        place=place.strip() if isinstance(place, str) and place.strip() else None,
        semantic_query=str(raw.get("semantic_query") or message),
        wants_all=bool(raw.get("wants_all")),
    )


def _entity_episode_ids(session: Session, user_id: str, names: list[str]) -> set:
    """Episode ids linked to any of the named entities (case-insensitive). Empty set = the user
    named an entity we don't know about."""
    lowered = [n.lower() for n in names]
    rows = session.exec(
        select(EpisodeEntity.episode_id)
        .join(Entity, col(EpisodeEntity.entity_id) == col(Entity.id))
        .where(Entity.user_id == user_id, func.lower(Entity.name).in_(lowered))
    ).all()
    return set(rows)


def _filtered_photos(
    session: Session, client, settings: Settings, *, user_id: str, spec: QuerySpec
) -> list[Episode]:
    stmt = select(Episode).where(
        Episode.user_id == user_id,
        col(Episode.context)["source"].astext == "image",  # pyrefly: ignore[missing-attribute]
    )
    if spec.time_range:
        start = datetime.combine(spec.time_range[0], datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(spec.time_range[1], datetime.max.time(), tzinfo=UTC)
        stmt = stmt.where(Episode.occurred_at >= start, Episode.occurred_at <= end)
    if spec.place:
        # pyrefly: ignore[missing-attribute]  — JSONB path -> text, case-insensitive contains
        stmt = stmt.where(col(Episode.context)["place"]["name"].astext.ilike(f"%{spec.place}%"))
    if spec.entities:
        ids = _entity_episode_ids(session, user_id, spec.entities)
        if not ids:
            return []  # asked about a named entity we don't know → honestly nothing, not a guess
        stmt = stmt.where(col(Episode.id).in_(ids))

    if spec.wants_all:
        # aggregate: the COMPLETE filtered set (capped for safety), newest first
        stmt = stmt.order_by(col(Episode.occurred_at).desc()).limit(FILTERED_CAP)
        return list(session.exec(stmt).all())

    # semantic: rank the filtered set by similarity to the rewritten query
    embedding = embed_text(client, settings.embedding_model, spec.semantic_query)
    stmt = (
        stmt.where(col(Episode.embedding).is_not(None))
        # pyrefly: ignore[missing-attribute]  — pgvector comparator missing from stubs
        .order_by(col(Episode.embedding).cosine_distance(embedding))
        .limit(RANKED_TOP_K)
    )
    return list(session.exec(stmt).all())


def retrieve(
    session: Session, client, settings: Settings, *, user_id: str, message: str
) -> RetrievalResult:
    """Decompose the question, filter the photo store exactly when it names an entity/time/place,
    else fall back to plain cosine. Facts are always searched by similarity, as before."""
    spec = decompose_query(client, settings, message)
    facts = search_memories(
        session, client, settings, user_id=user_id, query=message, limit=FACT_TOP_K
    )
    if spec.has_filters:
        photos = _filtered_photos(session, client, settings, user_id=user_id, spec=spec)
    else:
        photos = search_image_episodes(
            session, client, settings, user_id=user_id, query=message, limit=3
        )
    return RetrievalResult(facts=facts, photos=photos, spec=spec)