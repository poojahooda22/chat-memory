"""EXIF extraction for uploaded images.

Temporal and geographic context come from the file's own metadata, never from the vision
model — the model describes what it sees; the metadata says when and where. Both are
best-effort: screenshots carry neither, and images that passed through social apps arrive
stripped, so every field degrades gracefully and records which source was used.

Pillow note: read nested IFDs via getexif().get_ifd(...). The flat exif["GPSInfo"] returns
an int offset (not a dict) and must not be used.
"""

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime

from PIL import ExifTags, Image

DATETIME_ORIGINAL = 0x9003  # 36867 — capture time, in the Exif sub-IFD


@dataclass
class ExifInfo:
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    camera: str | None = None
    width: int = 0
    height: int = 0
    time_source: str = "upload"  # "exif" when captured_at came from the file
    raw: dict = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def as_context(self) -> dict:
        """The exif block stored on the job and inside Episode.context."""
        return {
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "time_source": self.time_source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
        }


def _to_degrees(value) -> float | None:
    """GPS coordinates are three rationals (degrees, minutes, seconds) -> decimal degrees."""
    try:
        d, m, s = (float(v) for v in value)
        return d + m / 60.0 + s / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _parse_datetime(value) -> datetime | None:
    """EXIF datetimes are 'YYYY:MM:DD HH:MM:SS' and timezone-naive; we normalise to UTC and
    flag the source so the UI can show 'captured' vs 'uploaded' honestly."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_exif(image_bytes: bytes) -> ExifInfo:
    info = ExifInfo()
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return info  # not an image Pillow can read; the route already checked content-type
    info.width, info.height = img.size

    exif = img.getexif()
    make = exif.get(0x010F)  # Make / Model live in the top-level IFD
    model = exif.get(0x0110)
    if make or model:
        info.camera = " ".join(str(p).strip() for p in (make, model) if p)

    captured = _parse_datetime(exif.get_ifd(ExifTags.IFD.Exif).get(DATETIME_ORIGINAL))
    if captured:
        info.captured_at = captured
        info.time_source = "exif"

    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps:
        lat = _to_degrees(gps.get(2))  # GPSLatitude
        lon = _to_degrees(gps.get(4))  # GPSLongitude
        if lat is not None and lon is not None:
            if str(gps.get(1, "N")).upper() == "S":  # GPSLatitudeRef
                lat = -lat
            if str(gps.get(3, "E")).upper() == "W":  # GPSLongitudeRef
                lon = -lon
            info.latitude = round(lat, 6)
            info.longitude = round(lon, 6)
    return info


def classify_kind(info: ExifInfo, content_type: str) -> str:
    """Screenshot vs photo: a screenshot is compositor output — no capture time, no GPS,
    no camera tags. A camera photo stripped by a social app can look the same, so the vision
    model's own reading later refines this; metadata alone is the first guess."""
    if info.captured_at is None and not info.has_gps and info.camera is None:
        return "screenshot"
    return "photo"