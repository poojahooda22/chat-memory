"""Entity labeling logic: the user names what the vision model detected.

"This is Monty" turns a detected "a golden retriever" into a named Entity. The label is new
knowledge arriving after the photo happened, so it never rewrites the episode (single-shot
invariant) — it creates/reuses an Entity row, links it to the episode, and records the fact
through the memory pipeline's decision phase (which deduplicates repeat labels). The links
are the co-occurrence substrate the relationship graph reads.

No commits here — the caller (route) owns the transaction, same as the rest of the pipeline.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlmodel import Session, col, func, select

from app.config import Settings
from app.memory.embeddings import embed_text
from app.memory.pipeline import record_fact
from app.models import Entity, Episode, EpisodeEntity


class LabelError(ValueError):
    """Raised when the episode or the detected-entity slot doesn't exist."""


# How close a detected description must be to a labeled entity before we SUGGEST its name.
# Cosine distance over text-embedding-3-small; tuned against real data (a matching pet pair
# lands well under this, an unrelated person/object well over). A suggestion is never an
# auto-label: the user confirms identity — the system only proposes.
SUGGEST_MAX_DISTANCE = 0.55


def suggest_entity(
    session: Session,
    client,
    settings: Settings,
    *,
    user_id: str,
    entity_type: str,
    description: str,
) -> tuple[Entity, float] | None:
    """The recognition step without biometrics: does this detected person/pet DESCRIPTION
    look like an entity the user already named? Embeds the description and cosine-matches
    against same-type labeled entities (HNSW index). Returns (entity, distance) when the
    best match is close enough, else None."""
    if not description.strip():
        return None
    embedding = embed_text(client, settings.embedding_model, description)
    stmt = (
        select(
            Entity,
            # pyrefly: ignore[missing-attribute]  — pgvector comparator missing from stubs
            col(Entity.embedding).cosine_distance(embedding).label("distance"),
        )
        .where(
            Entity.user_id == user_id,
            Entity.type == entity_type,
            col(Entity.embedding).is_not(None),
        )
        # pyrefly: ignore[missing-attribute]
        .order_by(col(Entity.embedding).cosine_distance(embedding))
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None
    entity, distance = row
    return (entity, float(distance)) if distance <= SUGGEST_MAX_DISTANCE else None


@dataclass
class LabelResult:
    entity: Entity
    memory_event: str  # ADD | UPDATE | DELETE | NOOP
    reused_existing: bool


_KIND_PHRASE = {"person": "the user's", "pet": "the user's pet", "object": "the user's"}


def apply_label(
    session: Session,
    client,
    settings: Settings,
    *,
    episode_id: uuid.UUID,
    entity_index: int,
    name: str,
) -> LabelResult:
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise LabelError("Episode not found")
    detected = (episode.context or {}).get("entities") or []
    if not (0 <= entity_index < len(detected)):
        raise LabelError(f"entity_index {entity_index} out of range")
    chip = detected[entity_index]

    name = name.strip()
    entity_type = chip.get("type", "object")
    description = chip.get("description", "")

    # find-or-create by (user, name, type), case-insensitive — "monty" and "Monty" are one
    existing = session.exec(
        select(Entity).where(
            Entity.user_id == episode.user_id,
            func.lower(Entity.name) == name.lower(),
            Entity.type == entity_type,
        )
    ).first()
    if existing is not None:
        entity = existing
        entity.updated_at = datetime.now(UTC)
    else:
        entity = Entity(
            user_id=episode.user_id,
            name=name,
            type=entity_type,
            description=description,
            embedding=embed_text(client, settings.embedding_model, f"{name}: {description}"),
        )
    session.add(entity)
    session.flush()

    # link once per (episode, entity, slot) — re-labeling the same chip adds nothing
    link = session.exec(
        select(EpisodeEntity).where(
            EpisodeEntity.episode_id == episode.id,
            EpisodeEntity.entity_id == entity.id,
            EpisodeEntity.entity_index == entity_index,
        )
    ).first()
    if link is None:
        session.add(
            EpisodeEntity(episode_id=episode.id, entity_id=entity.id, entity_index=entity_index)
        )

    # the label IS a fact — through the decision phase so repeats dedupe, with provenance
    kind_phrase = _KIND_PHRASE.get(entity_type, "the user's")
    fact = (
        f"{name} is {kind_phrase} {entity_type}: {description}"
        if description
        else f"{name} is {kind_phrase} {entity_type}"
    )
    op = record_fact(
        session, client, settings,
        user_id=episode.user_id, fact=fact, source_ids=[str(episode.id)],
    )
    return LabelResult(entity=entity, memory_event=op.event, reused_existing=existing is not None)