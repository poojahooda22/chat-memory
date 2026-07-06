"""Worker path: turn one uploaded image into an Episode + distilled memories.

The upload route wrote the job row + the file and returned 202; this runs off the request
path (BackgroundTasks, same precedent as the summary refresher). One transaction per job:
claim the queued row, annotate, INSERT the Episode exactly once, distil, mark done — so a
crash mid-work leaves the row 'queued' (re-runnable), never a half-written episode. The
episode_id on the job is the idempotency guard: a job that already produced an episode can
never insert a second one.
"""

from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session

from app.config import Settings
from app.ingest.vision import annotate_image
from app.memory.embeddings import embed_text
from app.memory.pipeline import distil_text
from app.models import Episode, IngestJob


def run_ingest_job(engine, client, settings: Settings, job_id) -> None:
    """Background entry point — own session, own commit; failures land on the job row."""
    with Session(engine) as session:
        try:
            process_job(session, client, settings, job_id=job_id)
            session.commit()
        except Exception as exc:
            session.rollback()
            _mark_failed(session, job_id, exc)


def _mark_failed(session: Session, job_id, exc: Exception) -> None:
    """Record the failure on the job row — the API surfaces it; nothing is swallowed."""
    job = session.get(IngestJob, job_id)
    if job is None:
        return
    job.status = "failed"
    job.error = f"{type(exc).__name__}: {exc}"[:2000]
    job.updated_at = datetime.now(UTC)
    session.add(job)
    session.commit()


def _entities_from(annotation: dict) -> list[dict]:
    """Flatten the vision annotation into context.entities — labels stay None until the
    user names them (the labeling step); descriptions are what the model actually saw."""
    entities: list[dict] = []
    for p in annotation.get("people") or []:
        if isinstance(p, dict) and p.get("description"):
            entities.append({
                "type": "person", "description": str(p["description"]).strip(),
                "label": None, "confidence": _confidence(p),
            })
    for p in annotation.get("pets") or []:
        if isinstance(p, dict) and p.get("description"):
            entities.append({
                "type": "pet", "description": str(p["description"]).strip(),
                "species": str(p.get("species") or "").strip() or None,
                "label": None, "confidence": _confidence(p),
            })
    for o in annotation.get("objects") or []:
        if isinstance(o, str) and o.strip():
            entities.append({"type": "object", "description": o.strip(), "label": None})
    return entities


def _confidence(item: dict) -> float:
    try:
        return max(0.0, min(1.0, float(item.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def process_job(session: Session, client, settings: Settings, *, job_id) -> IngestJob | None:
    job = session.get(IngestJob, job_id)
    # status guard: only a queued job runs; done/failed need an explicit retry
    if job is None or job.status != "queued":
        return job
    # idempotency backstop: an episode already exists -> just finish the bookkeeping
    if job.episode_id is not None:
        job.status = "done"
        session.add(job)
        return job

    job.status = "processing"
    session.add(job)

    image_bytes = Path(job.image_path).read_bytes()
    annotation = annotate_image(client, settings.vision_model, image_bytes, job.content_type)

    # the model's own reading refines the metadata-only first guess (a stripped photo looks
    # like a screenshot to EXIF; the pixels disambiguate)
    if annotation.get("kind") in ("photo", "screenshot"):
        job.kind = annotation["kind"]

    caption = str(annotation.get("caption") or "").strip() or f"an uploaded {job.kind}"
    ocr = str(annotation.get("ocr_text") or "").strip()
    content = f"{caption}\n\nText in the image:\n{ocr}" if ocr else caption

    exif = job.exif or {}
    occurred_at = job.created_at  # fallback: upload time
    if exif.get("captured_at"):
        occurred_at = datetime.fromisoformat(exif["captured_at"])

    episode = Episode(
        user_id=job.user_id,
        conversation_id=None,
        occurred_at=occurred_at,
        content=content,
        context={
            "source": "image",
            "kind": job.kind,
            "entities": _entities_from(annotation),
            "environment": str(annotation.get("environment") or ""),
            "activity": str(annotation.get("activity") or ""),
            "emotion": str(annotation.get("emotion") or ""),
            "place": {
                "latitude": exif.get("latitude"),
                "longitude": exif.get("longitude"),
                "guess": annotation.get("place_guess"),
            },
            "exif": exif,
            "ingest_job_id": str(job.id),
        },
        embedding=embed_text(client, settings.embedding_model, content),
    )
    session.add(episode)
    session.flush()  # assign the id before it goes onto the job

    job.episode_id = episode.id

    # the caption flows through the unchanged Mem0 extract->decide pipeline, so a photo
    # produces semantic memories exactly like a chat message does. Framed as life content,
    # not as an upload event — "I added a screenshot" is not a durable fact.
    distil_text(
        session, client, settings,
        user_id=job.user_id,
        text=f"A {job.kind} from my life shows: {content}",
        source_ids=[str(episode.id)],
    )

    job.status = "done"
    job.error = None
    job.updated_at = datetime.now(UTC)
    session.add(job)
    return job