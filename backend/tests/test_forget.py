"""Forgetting an upload cascades honestly: file + job + episode + links go; a memory keeps
living if other episodes still support it, and is forgotten (with an audit row) if not."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import select

from app.ingest.forget import forget_job
from app.models import Entity, Episode, EpisodeEntity, IngestJob, Memory, MemoryHistory
from tests.conftest import fake_embedding


def _seed(db_session, tmp_path, user_id: str):
    """One photo episode + its job + a labeled entity link + two memories:
    one supported ONLY by this photo, one also supported by another episode."""
    episode = Episode(
        user_id=user_id,
        occurred_at=datetime(2022, 7, 15, 18, 30, tzinfo=UTC),
        content="A blue circle with the text SMOKE TEST.",
        context={"source": "image", "kind": "screenshot",
                 "entities": [{"type": "pet", "description": "a dog", "label": None}]},
        embedding=fake_embedding("smoke test image"),
    )
    other = Episode(
        user_id=user_id, occurred_at=datetime(2023, 5, 22, tzinfo=UTC),
        content="chat about the dog", context={"role": "user", "source": "chat"},
        embedding=fake_embedding("chat about the dog"),
    )
    db_session.add(episode)
    db_session.add(other)
    db_session.flush()

    image = tmp_path / "smoke.jpg"
    image.write_bytes(b"fake-jpeg-bytes")
    job = IngestJob(
        user_id=user_id, kind="screenshot", status="done", filename="smoke.jpg",
        content_type="image/jpeg", image_path=str(image), exif={}, episode_id=episode.id,
    )
    entity = Entity(user_id=user_id, name="Testo", type="pet", description="a dog")
    db_session.add(job)
    db_session.add(entity)
    db_session.flush()
    db_session.add(EpisodeEntity(episode_id=episode.id, entity_id=entity.id, entity_index=0))

    only_this = Memory(
        user_id=user_id, content="Fact only from the smoke photo",
        embedding=fake_embedding("Fact only from the smoke photo"),
        source_episode_ids=[str(episode.id)],
    )
    shared = Memory(
        user_id=user_id, content="Has a dog",
        embedding=fake_embedding("Has a dog"),
        source_episode_ids=[str(episode.id), str(other.id)],
    )
    db_session.add(only_this)
    db_session.add(shared)
    db_session.flush()
    return job, episode, entity, only_this, shared, image


def test_forget_job_cascades(db_session, tmp_path):
    user = f"test-{uuid.uuid4()}"
    job, episode, entity, only_this, shared, image = _seed(db_session, tmp_path, user)

    assert forget_job(db_session, job_id=job.id) is True
    db_session.flush()

    # job, episode, links, file: gone
    assert db_session.get(IngestJob, job.id) is None
    assert db_session.get(Episode, episode.id) is None
    assert not list(db_session.exec(
        select(EpisodeEntity).where(EpisodeEntity.episode_id == episode.id)).all())
    assert not image.exists()
    # the single-source memory was forgotten, with an audit row
    db_session.refresh(only_this)
    assert only_this.is_deleted is True
    events = list(db_session.exec(
        select(MemoryHistory).where(MemoryHistory.memory_id == only_this.id)).all())
    assert any(h.event == "DELETE" for h in events)
    # the shared memory lives on, minus this receipt
    db_session.refresh(shared)
    assert shared.is_deleted is False
    assert shared.source_episode_ids == [s for s in shared.source_episode_ids if s != str(episode.id)]
    assert len(shared.source_episode_ids) == 1
    # the entity survives — knowledge outlives any single photo
    assert db_session.get(Entity, entity.id) is not None


def test_forget_unknown_job_returns_false(db_session):
    assert forget_job(db_session, job_id=uuid.uuid4()) is False