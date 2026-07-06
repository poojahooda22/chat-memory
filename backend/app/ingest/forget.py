"""Forgetting an uploaded image — the right-to-forget applied to one photo.

Deleting a source cascades honestly: the stored file, the ingest job, the episode and its
entity links all go; distilled memories are handled by provenance — a memory whose ONLY
source was this episode is soft-deleted (with an audit row, reversible in the DB), while a
memory with other sources just loses this episode from its receipts. Entities themselves
survive: knowledge like "Monty is the user's pet" outlives any single photo.

No commits here — the route owns the transaction.
"""

import uuid
from pathlib import Path

from sqlmodel import Session, col, select

from app.models import Episode, EpisodeEntity, IngestJob, Memory, MemoryHistory


def forget_job(session: Session, *, job_id: uuid.UUID) -> bool:
    """Remove one upload and everything only it supported. Returns False if unknown."""
    job = session.get(IngestJob, job_id)
    if job is None:
        return False
    image_path = job.image_path
    episode_id = job.episode_id

    # the job row goes first (it holds the FK onto the episode)
    session.delete(job)

    if episode_id is not None:
        eid = str(episode_id)

        # entity links for this episode (the entity rows themselves stay)
        for link in session.exec(
            select(EpisodeEntity).where(EpisodeEntity.episode_id == episode_id)
        ).all():
            session.delete(link)

        # memories that cite this episode: strip the receipt, or forget if it was the only one
        cited = session.exec(
            select(Memory).where(
                Memory.is_deleted == False,  # noqa: E712
                col(Memory.source_episode_ids).contains([eid]),
            )
        ).all()
        for memory in cited:
            remaining = [s for s in memory.source_episode_ids if s != eid]
            if remaining:
                memory.source_episode_ids = remaining
                session.add(memory)
            else:
                memory.is_deleted = True
                session.add(memory)
                session.add(
                    MemoryHistory(
                        memory_id=memory.id, event="DELETE", old_content=memory.content
                    )
                )

        episode = session.get(Episode, episode_id)
        if episode is not None:
            session.delete(episode)

    # the original bytes last — DB cleanup matters more than a locked file on Windows
    if image_path:
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass  # orphaned file is harmless; the DB no longer references it

    return True