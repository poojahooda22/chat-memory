"""The mouth: images enter memory here.

POST /uploads does only cheap synchronous work — persist the original bytes, parse EXIF,
write the queued job row — and returns 202; the vision call runs off the request path.
Files are stored under the job's own id (never the client filename), and served back by id
lookup only.
"""

import mimetypes
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.config import get_settings
from app.db import get_session
from app.ingest.exif import classify_kind, parse_exif
from app.ingest.pipeline import run_ingest_job
from app.models import Episode, IngestJob

router = APIRouter(tags=["uploads"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # a 20MB original is already generous for a photo


class IngestJobRead(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    filename: str
    captured_at: str | None
    time_source: str | None
    episode_id: uuid.UUID | None
    caption: str | None
    error: str | None
    created_at: datetime


def _job_out(job: IngestJob, caption: str | None) -> IngestJobRead:
    return IngestJobRead(
        id=job.id,
        kind=job.kind,
        status=job.status,
        filename=job.filename,
        captured_at=(job.exif or {}).get("captured_at"),
        time_source=(job.exif or {}).get("time_source"),
        episode_id=job.episode_id,
        caption=caption,
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
    user_id: str = Form("default"),
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


@router.get("/uploads", operation_id="list_uploads", response_model=list[IngestJobRead])
def list_uploads(
    user_id: str = "default", session: Session = Depends(get_session)
) -> list[IngestJobRead]:
    stmt = (
        select(IngestJob)
        .where(IngestJob.user_id == user_id)
        .order_by(col(IngestJob.created_at).desc())
        .limit(200)
    )
    jobs = list(session.exec(stmt).all())

    # one query for the captions of every finished job
    episode_ids = [j.episode_id for j in jobs if j.episode_id is not None]
    captions: dict[uuid.UUID, str] = {}
    if episode_ids:
        for e in session.exec(select(Episode).where(col(Episode.id).in_(episode_ids))).all():
            captions[e.id] = e.content
    return [_job_out(j, captions.get(j.episode_id) if j.episode_id else None) for j in jobs]


@router.get("/uploads/{job_id}/image", operation_id="upload_image_file")
def upload_image_file(job_id: uuid.UUID, session: Session = Depends(get_session)) -> FileResponse:
    job = session.get(IngestJob, job_id)
    if job is None or not job.image_path or not Path(job.image_path).is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(job.image_path, media_type=job.content_type)


@router.post(
    "/uploads/{job_id}/retry", operation_id="retry_upload", status_code=202,
    response_model=IngestJobRead,
)
def retry_upload(
    job_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> IngestJobRead:
    """Re-kick a failed job, or a queued one orphaned by a server restart."""
    job = session.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(404, "Upload not found")
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