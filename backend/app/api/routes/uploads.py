"""The mouth: images enter memory here.

POST /uploads does only cheap synchronous work — persist the original bytes, parse EXIF,
write the queued job row — and returns 202; the vision call runs off the request path.
Files are stored under the job's own id (never the client filename), and served back by id
lookup only.
"""

import mimetypes
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.auth import get_current_user
from app.config import get_settings
from app.db import get_session
from app.ingest.exif import classify_kind, parse_exif
from app.ingest.forget import forget_job
from app.ingest.pipeline import run_ingest_job
from app.ingest.takeout import build_jobs_from_zip
from app.memory.entities import suggest_entity
from app.models import Entity, Episode, EpisodeEntity, IngestJob


def _owned_job(session: Session, job_id: uuid.UUID, user_id: str) -> IngestJob:
    """Fetch a job that belongs to this user, or 404 — right user, right row."""
    job = session.get(IngestJob, job_id)
    if job is None or job.user_id != user_id:
        raise HTTPException(404, "Upload not found")
    return job

router = APIRouter(tags=["uploads"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # a 20MB original is already generous for a photo


class EntityChip(BaseModel):
    """One detected entity on a photo, with the user's label once it has one."""

    index: int
    type: str
    description: str
    confidence: float | None = None
    label: str | None = None
    labeled_by: str | None = None  # "user" | "memory" (visual recognition)
    # recognition: "looks like an entity you already named" — a proposal the user confirms
    suggested_name: str | None = None


class IngestJobRead(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    filename: str
    captured_at: str | None
    time_source: str | None
    place: str | None = None  # geocoded place name, when the photo carried GPS
    episode_id: uuid.UUID | None
    caption: str | None
    entities: list[EntityChip] = []
    error: str | None
    created_at: datetime


def _job_out(
    job: IngestJob,
    caption: str | None,
    entities: list[EntityChip] | None = None,
    place: str | None = None,
) -> IngestJobRead:
    return IngestJobRead(
        id=job.id,
        kind=job.kind,
        status=job.status,
        filename=job.filename,
        captured_at=(job.exif or {}).get("captured_at"),
        time_source=(job.exif or {}).get("time_source"),
        place=place,
        episode_id=job.episode_id,
        caption=caption,
        entities=entities or [],
        error=job.error,
        created_at=job.created_at,
    )


@router.post(
    "/uploads", operation_id="upload_images", status_code=202,
    response_model=list[IngestJobRead],
)
def upload_images(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[IngestJobRead]:
    settings = get_settings()
    uploads_dir = Path(settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[IngestJob] = []
    for f in files:
        content_type = f.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(415, f"'{f.filename}' is {content_type or 'unknown'}, not an image")
        data = f.file.read()
        if not data:
            raise HTTPException(422, f"'{f.filename}' is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"'{f.filename}' exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")

        info = parse_exif(data)
        job = IngestJob(
            user_id=user_id,
            kind=classify_kind(info, content_type),
            filename=f.filename or "",
            content_type=content_type,
            image_path="",
            exif=info.as_context(),
        )
        # stored under the job id, never the client filename — no path traversal surface
        path = uploads_dir / f"{job.id}{mimetypes.guess_extension(content_type) or '.bin'}"
        path.write_bytes(data)
        job.image_path = str(path)
        session.add(job)
        jobs.append(job)

    session.commit()
    for job in jobs:
        background.add_task(
            run_ingest_job,
            request.app.state.engine, request.app.state.llm, settings, job.id,
        )
    return [_job_out(job, caption=None) for job in jobs]


MAX_TAKEOUT_BYTES = 500 * 1024 * 1024  # 500MB in-memory ceiling (Tier-2 = streaming extraction)


@router.post(
    "/uploads/takeout", operation_id="import_takeout", status_code=202,
    response_model=list[IngestJobRead],
)
def import_takeout(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[IngestJobRead]:
    """Import a Google Takeout .zip. Each photo's JSON sidecar restores the GPS + capture time
    Google strips on download, so photos come in WITH locations. Reuses the image-ingest worker:
    the same queued jobs an ordinary upload produces, geocoded + distilled off the request path."""
    settings = get_settings()
    data = file.file.read()
    if not data:
        raise HTTPException(422, "The export file is empty")
    if len(data) > MAX_TAKEOUT_BYTES:
        raise HTTPException(413, "Export too large; choose a smaller split size in Google Takeout")
    try:
        jobs = build_jobs_from_zip(session, settings, user_id=user_id, zip_bytes=data)
    except zipfile.BadZipFile:
        raise HTTPException(415, "That doesn't look like a valid .zip export") from None
    if not jobs:
        raise HTTPException(422, "No photos found — expected a Google Photos Takeout .zip")

    session.commit()
    for job in jobs:
        background.add_task(
            run_ingest_job,
            request.app.state.engine, request.app.state.llm, settings, job.id,
        )
    return [_job_out(job, caption=None) for job in jobs]


@router.get("/uploads", operation_id="list_uploads", response_model=list[IngestJobRead])
def list_uploads(
    request: Request,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[IngestJobRead]:
    stmt = (
        select(IngestJob)
        .where(IngestJob.user_id == user_id)
        .order_by(col(IngestJob.created_at).desc())
        .limit(200)
    )
    jobs = list(session.exec(stmt).all())

    # one query for the episodes of every finished job, one for their entity labels
    episode_ids = [j.episode_id for j in jobs if j.episode_id is not None]
    episodes: dict[uuid.UUID, Episode] = {}
    labels: dict[uuid.UUID, dict[int, tuple[str, str]]] = {}  # episode -> {index: (name, by)}
    if episode_ids:
        for e in session.exec(select(Episode).where(col(Episode.id).in_(episode_ids))).all():
            episodes[e.id] = e
        rows = session.exec(
            select(EpisodeEntity, Entity)
            .join(Entity, col(EpisodeEntity.entity_id) == col(Entity.id))
            .where(col(EpisodeEntity.episode_id).in_(episode_ids))
        ).all()
        for link, entity in rows:
            labels.setdefault(link.episode_id, {})[link.entity_index] = (
                entity.name,
                link.labeled_by,
            )

    # recognition memo: identical descriptions across photos embed + match only once
    settings = get_settings()
    suggestion_cache: dict[tuple[str, str], str | None] = {}

    def _chips(episode: Episode) -> list[EntityChip]:
        named = labels.get(episode.id, {})
        chips: list[EntityChip] = []
        for i, d in enumerate((episode.context or {}).get("entities") or []):
            entity_type = d.get("type", "object")
            description = d.get("description", "")
            pair = named.get(i)  # (name, labeled_by) | None
            suggested = None
            # only unlabeled people/pets get a recognition pass
            if pair is None and entity_type in ("person", "pet"):
                key = (entity_type, description)
                if key not in suggestion_cache:
                    match = suggest_entity(
                        session, request.app.state.llm, settings,
                        user_id=user_id, entity_type=entity_type, description=description,
                    )
                    suggestion_cache[key] = match[0].name if match else None
                suggested = suggestion_cache[key]
            chips.append(
                EntityChip(
                    index=i,
                    type=entity_type,
                    description=description,
                    confidence=d.get("confidence"),
                    label=pair[0] if pair else None,
                    labeled_by=pair[1] if pair else None,
                    suggested_name=suggested,
                )
            )
        return chips

    out: list[IngestJobRead] = []
    for j in jobs:
        episode = episodes.get(j.episode_id) if j.episode_id else None
        place = ((episode.context or {}).get("place") or {}).get("name") if episode else None
        out.append(
            _job_out(
                j,
                caption=episode.content if episode else None,
                entities=_chips(episode) if episode else [],
                place=place,
            )
        )
    return out


@router.get("/uploads/{job_id}/image", operation_id="upload_image_file")
def upload_image_file(
    job_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> FileResponse:
    job = _owned_job(session, job_id, user_id)
    if not job.image_path or not Path(job.image_path).is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(job.image_path, media_type=job.content_type)


class RenameRequest(BaseModel):
    filename: str


@router.patch("/uploads/{job_id}", operation_id="rename_upload", response_model=IngestJobRead)
def rename_upload(
    job_id: uuid.UUID,
    req: RenameRequest,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> IngestJobRead:
    """Give the upload a human name — Drive/Photos downloads arrive as IMG20230522… noise."""
    job = _owned_job(session, job_id, user_id)
    name = req.filename.strip()
    if not name:
        raise HTTPException(422, "filename must not be empty")
    job.filename = name[:200]
    session.add(job)
    session.commit()
    return _job_out(job, caption=None)


@router.delete("/uploads/{job_id}", operation_id="delete_upload")
def delete_upload(
    job_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """Forget this photo: file + episode + links go; memories lose this receipt (or are
    forgotten when it was their only source). Entities survive."""
    _owned_job(session, job_id, user_id)  # 404 unless it's this user's photo
    forget_job(session, job_id=job_id)
    session.commit()
    return {"status": "deleted", "job_id": str(job_id)}


@router.post(
    "/uploads/{job_id}/retry", operation_id="retry_upload", status_code=202,
    response_model=IngestJobRead,
)
def retry_upload(
    job_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> IngestJobRead:
    """Re-kick a failed job, or a queued one orphaned by a server restart."""
    job = _owned_job(session, job_id, user_id)
    if job.status not in ("queued", "failed"):
        raise HTTPException(409, f"Job is {job.status}; only queued/failed jobs can be retried")
    job.status = "queued"
    job.error = None
    session.add(job)
    session.commit()
    background.add_task(
        run_ingest_job,
        request.app.state.engine, request.app.state.llm, get_settings(), job.id,
    )
    return _job_out(job, caption=None)