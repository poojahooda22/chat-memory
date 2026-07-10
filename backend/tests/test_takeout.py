"""The Google Takeout sidecar parser: recover the GPS + capture time Google strips from the
downloaded file and keeps in the JSON sidecar. Pure functions — no network, no DB."""

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.ingest.takeout import (
    build_jobs_from_zip,
    exif_from_takeout,
    find_image_sidecar_pairs,
    pair_sidecar,
    parse_sidecar,
)


def _tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 8), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _sidecar(lat=28.61, lon=77.20, ts="1646742855", block="geoData") -> bytes:
    return json.dumps({
        "title": "IMG.jpg",
        "photoTakenTime": {"timestamp": ts},
        block: {"latitude": lat, "longitude": lon, "altitude": 0.0},
    }).encode()


def test_parse_sidecar_reads_geodata_and_time():
    lat, lon, when = parse_sidecar(_sidecar())
    assert lat == 28.61 and lon == 77.2
    assert when is not None and when.year == 2022  # 1646742855 -> March 2022


def test_parse_sidecar_zero_gps_is_treated_as_absent():
    lat, lon, _ = parse_sidecar(_sidecar(lat=0.0, lon=0.0))
    assert lat is None and lon is None  # Google writes 0.0 for "no location" — never a fake (0,0)


def test_parse_sidecar_falls_back_to_geodataexif():
    raw = json.dumps({
        "geoData": {"latitude": 0.0, "longitude": 0.0},
        "geoDataExif": {"latitude": 12.9, "longitude": 77.6},
    }).encode()
    lat, lon, _ = parse_sidecar(raw)
    assert lat == 12.9 and lon == 77.6


def test_parse_sidecar_malformed_returns_nones():
    assert parse_sidecar(b"not json at all") == (None, None, None)
    assert parse_sidecar(b"[]") == (None, None, None)  # valid json, wrong shape


def test_exif_from_takeout_overlays_gps_onto_stripped_image():
    info = exif_from_takeout(_tiny_jpeg(), _sidecar(lat=28.61, lon=77.2))
    assert info.latitude == 28.61 and info.longitude == 77.2  # recovered from the sidecar
    assert info.time_source == "takeout"
    assert info.width == 10 and info.height == 8  # dimensions still read from the image itself
    assert info.as_context()["latitude"] == 28.61  # flows into the job.exif shape the worker reads


def test_exif_from_takeout_without_sidecar_is_just_the_image():
    info = exif_from_takeout(_tiny_jpeg(), None)
    assert info.latitude is None and info.longitude is None  # no sidecar -> no recovered location


def test_pairing_matches_both_sidecar_conventions():
    names = [
        "IMG_1.jpg", "IMG_1.jpg.json",
        "IMG_2.jpg", "IMG_2.jpg.supplemental-metadata.json",
        "IMG_3.jpg",  # no sidecar
    ]
    assert pair_sidecar("IMG_1.jpg", set(names)) == "IMG_1.jpg.json"
    assert pair_sidecar("IMG_2.jpg", set(names)) == "IMG_2.jpg.supplemental-metadata.json"
    assert pair_sidecar("IMG_3.jpg", set(names)) is None

    pairs = dict(find_image_sidecar_pairs(names))
    assert set(pairs) == {"IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"}  # one entry per image
    assert pairs["IMG_1.jpg"] == "IMG_1.jpg.json"
    assert pairs["IMG_3.jpg"] is None  # imports anyway, just without location


def test_build_jobs_from_zip_creates_jobs_with_sidecar_gps(db_session, tmp_path):
    """A real (in-memory) Takeout zip becomes queued ingest jobs — the sidecar's GPS lands on
    job.exif (what the worker geocodes), and a photo without a sidecar still imports."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Takeout/Google Photos/Album/IMG_1.jpg", _tiny_jpeg())
        zf.writestr("Takeout/Google Photos/Album/IMG_1.jpg.json", _sidecar(lat=28.61, lon=77.2))
        zf.writestr("Takeout/Google Photos/Album/IMG_2.jpg", _tiny_jpeg())  # no sidecar

    settings = get_settings().model_copy(update={"uploads_dir": str(tmp_path)})
    jobs = build_jobs_from_zip(db_session, settings, user_id="tk-test", zip_bytes=buf.getvalue())

    assert len(jobs) == 2
    by_name = {j.filename: j for j in jobs}
    assert by_name["IMG_1.jpg"].exif["latitude"] == 28.61  # recovered from the sidecar
    assert by_name["IMG_2.jpg"].exif["latitude"] is None  # no sidecar -> honestly no location
    assert by_name["IMG_1.jpg"].status == "queued"  # ready for the existing worker
    # the original bytes were saved under the job id (never the archive path)
    assert Path(by_name["IMG_1.jpg"].image_path).is_file()