"""Read a Google Takeout export — recovering the location Google strips on a normal download.

When you download a photo from Google Photos, the GPS is removed from the file's EXIF. Google
keeps it, and a TAKEOUT export hands it back in a per-photo JSON SIDECAR sitting next to each
image (`IMG_x.jpg` + `IMG_x.jpg.json` / `IMG_x.jpg.supplemental-metadata.json`), carrying
`geoData` (lat/lon) and `photoTakenTime`. This module reads that sidecar and OVERLAYS its GPS +
capture time onto the image's own EXIF — so the SAME ingest worker geocodes it, with no worker
changes (process_job already reads job.exif; we just fill it from the sidecar).

Sidecar shape (Google Photos Takeout):
    {"title": "...", "photoTakenTime": {"timestamp": "1646742855"},
     "geoData": {"latitude": 28.61, "longitude": 77.20, ...},
     "geoDataExif": {...}, "people": [...], "description": "..."}
When there is no location, Google writes latitude/longitude as 0.0 — treated here as absent.
"""

import io
import json
import mimetypes
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session

from app.config import Settings
from app.ingest.exif import ExifInfo, classify_kind, parse_exif
from app.models import IngestJob

# metadata-filename suffixes Google has used over the years. The truncated-long-name variant is a
# known gap (Google shortens the whole sidecar name for long titles) — refined later, flagged now.
_SIDECAR_SUFFIXES = (".supplemental-metadata.json", ".suppl.json", ".json")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")


def _coord(geo: dict, key: str) -> float | None:
    """A geoData coordinate, or None. Google writes exactly 0.0 for 'no location', so 0 is absent."""
    raw = geo.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return None if value == 0.0 else round(value, 6)


def parse_sidecar(sidecar_bytes: bytes) -> tuple[float | None, float | None, datetime | None]:
    """Pull (latitude, longitude, captured_at) from a Takeout sidecar.

    Prefers `geoData`, falls back to `geoDataExif`; capture time from `photoTakenTime`
    (unix-seconds) then `creationTime`. Any field may be missing -> None (never fabricated).
    """
    try:
        data = json.loads(sidecar_bytes)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    latitude = longitude = None
    for block in ("geoData", "geoDataExif"):
        geo = data.get(block)
        if isinstance(geo, dict):
            latitude, longitude = _coord(geo, "latitude"), _coord(geo, "longitude")
            if latitude is not None and longitude is not None:
                break

    captured_at = None
    taken = data.get("photoTakenTime") or data.get("creationTime")
    if isinstance(taken, dict) and taken.get("timestamp"):
        try:
            captured_at = datetime.fromtimestamp(int(taken["timestamp"]), tz=UTC)
        except (TypeError, ValueError, OSError, OverflowError):
            captured_at = None
    return latitude, longitude, captured_at


def exif_from_takeout(image_bytes: bytes, sidecar_bytes: bytes | None) -> ExifInfo:
    """The exif block for a Takeout photo: the image's own EXIF (dimensions, camera, any surviving
    capture time) with the sidecar's GPS + capture time OVERLAID as the authoritative source —
    Google's copy is the real location; the downloaded file was stripped."""
    info = parse_exif(image_bytes)  # width/height/camera (+ captured_at if the file still has it)
    if not sidecar_bytes:
        return info
    latitude, longitude, captured_at = parse_sidecar(sidecar_bytes)
    if latitude is not None and longitude is not None:
        info.latitude, info.longitude = latitude, longitude
    if captured_at is not None:
        info.captured_at = captured_at
        info.time_source = "takeout"
    return info


def _is_image(name: str) -> bool:
    return name.lower().endswith(_IMAGE_SUFFIXES)


def pair_sidecar(image_name: str, names: set[str]) -> str | None:
    """The sidecar filename for an image, among a set of names. Google's convention is the image
    name + a metadata suffix; over the years that has been `.json` and
    `.supplemental-metadata.json`."""
    for suffix in _SIDECAR_SUFFIXES:
        candidate = image_name + suffix
        if candidate in names:
            return candidate
    return None


def find_image_sidecar_pairs(names: list[str]) -> list[tuple[str, str | None]]:
    """Given every filename in a Takeout folder, return (image, sidecar-or-None) pairs — one per
    image. A missing sidecar is fine: the image still imports, just without recovered location."""
    nameset = set(names)
    return [(name, pair_sidecar(name, nameset)) for name in names if _is_image(name)]


IMPORT_CAP = 500  # max photos per import (first-version safety; Tier-2 = streaming + a real queue)
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # skip a single oversized entry rather than fail the whole import


def build_jobs_from_zip(
    session: Session, settings: Settings, *, user_id: str, zip_bytes: bytes
) -> list[IngestJob]:
    """Extract a Google Takeout ZIP into queued ingest jobs.

    For each image + sidecar pair: overlay the sidecar's GPS/time onto the image EXIF, save the
    original bytes under the job id, and create the SAME queued job row a normal upload produces —
    so the existing worker geocodes + distils each one, with no worker changes. Capped at
    IMPORT_CAP for a first version; a huge library (streaming extraction, a real queue) is Tier-2.
    """
    uploads_dir = Path(settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[IngestJob] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for image_name, sidecar_name in find_image_sidecar_pairs(archive.namelist()):
            if len(jobs) >= IMPORT_CAP:
                break
            image_bytes = archive.read(image_name)
            if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                continue
            sidecar_bytes = archive.read(sidecar_name) if sidecar_name else None
            info = exif_from_takeout(image_bytes, sidecar_bytes)
            content_type = mimetypes.guess_type(image_name)[0] or "image/jpeg"
            job = IngestJob(
                user_id=user_id,
                kind=classify_kind(info, content_type),
                filename=Path(image_name).name,  # basename only — never the folder path
                content_type=content_type,
                image_path="",
                exif=info.as_context(),
            )
            # stored under the job id, never the archive path — no path-traversal surface
            path = uploads_dir / f"{job.id}{mimetypes.guess_extension(content_type) or '.jpg'}"
            path.write_bytes(image_bytes)
            job.image_path = str(path)
            session.add(job)
            jobs.append(job)
    return jobs