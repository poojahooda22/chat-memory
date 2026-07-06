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

from sqlmodel import Session, col, select

from app.config import Settings
from app.ingest.vision import annotate_image, compare_subjects
from app.memory.embeddings import embed_text
from app.memory.entities import apply_label, suggest_entity
from app.memory.pipeline import distil_text
from app.models import Episode, EpisodeEntity, IngestJob

# visual recognition must be at least this sure before the memory labels a pet by itself
AUTO_LABEL_MIN_CONFIDENCE = 0.8


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

    # visual recognition: pets the memory already knows get their label applied by the
    # memory itself — the user taught the name once (single-shot); recognition is the
    # system's job from then on. People stay propose-only by design.
    try:
        _auto_label_pets(session, client, settings, job=job, episode=episode, image_bytes=image_bytes)
    except Exception as exc:  # recognition is an enhancement — its failure never fails ingest
        job.error = f"auto-label skipped: {type(exc).__name__}: {exc}"[:500]

    job.status = "done"
    job.updated_at = datetime.now(UTC)
    session.add(job)
    return job


def _auto_label_pets(
    session: Session, client, settings: Settings, *, job: IngestJob, episode: Episode, image_bytes: bytes
) -> None:
    for index, chip in enumerate((episode.context or {}).get("entities") or []):
        if chip.get("type") != "pet":
            continue
        candidate = suggest_entity(
            session, client, settings,
            user_id=job.user_id, entity_type="pet", description=chip.get("description", ""),
        )
        if candidate is None:
            continue
        entity, _distance = candidate
        reference = _reference_image(session, entity_id=entity.id, exclude_episode_id=episode.id)
        if reference is None:
            continue
        ref_bytes, ref_type = reference
        verdict = compare_subjects(
            client, settings.vision_model,
            reference_bytes=ref_bytes, reference_type=ref_type,
            candidate_bytes=image_bytes, candidate_type=job.content_type,
        )
        confidence = _confidence(verdict)
        if bool(verdict.get("same_subject")) and confidence >= AUTO_LABEL_MIN_CONFIDENCE:
            apply_label(
                session, client, settings,
                episode_id=episode.id, entity_index=index,
                name=entity.name, labeled_by="memory",
            )


def _reference_image(
    session: Session, *, entity_id, exclude_episode_id
) -> tuple[bytes, str] | None:
    """The most recent USER-confirmed photo of this entity that still has its file.

    Auto-labeled photos are deliberately never references — recognition always anchors on
    something the user personally confirmed, so a wrong auto-label cannot compound.
    """
    links = session.exec(
        select(EpisodeEntity)
        .where(
            EpisodeEntity.entity_id == entity_id,
            EpisodeEntity.episode_id != exclude_episode_id,
            EpisodeEntity.labeled_by == "user",
        )
        .order_by(col(EpisodeEntity.created_at).desc())
    ).all()
    for link in links:
        ref_job = session.exec(
            select(IngestJob).where(IngestJob.episode_id == link.episode_id)
        ).first()
        if ref_job and ref_job.image_path and Path(ref_job.image_path).is_file():
            return Path(ref_job.image_path).read_bytes(), ref_job.content_type
    return None