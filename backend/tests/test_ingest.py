"""Image ingest: EXIF parsing, the worker's episode+distil run, and its idempotency.

The vision + extraction + decision calls ride the FakeLLM (zero tokens); the EXIF tests run
against a real JPEG synthesized with Pillow so the parser is exercised on genuine bytes.
"""

import io
import json
from datetime import UTC, datetime

import pytest
from PIL import ExifTags, Image
from sqlmodel import select

from app.config import get_settings
from app.ingest.exif import classify_kind, parse_exif
from app.ingest.pipeline import process_job
from app.models import Episode, IngestJob, Memory
from tests.conftest import FakeLLM, fake_embedding

CAPTURE = "2022:07:15 18:30:00"


def _jpeg_with_exif() -> bytes:
    """A tiny real JPEG carrying DateTimeOriginal + GPS (28.6°N, 77.2°E) + camera tags."""
    img = Image.new("RGB", (64, 48), color=(200, 120, 40))
    exif = Image.Exif()
    exif[0x010F] = "TestMake"  # Make
    exif[0x0110] = "TestCam"  # Model
    exif.get_ifd(ExifTags.IFD.Exif)[0x9003] = CAPTURE  # DateTimeOriginal
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps[1] = "N"
    gps[2] = (28.0, 36.0, 0.0)  # 28°36'
    gps[3] = "E"
    gps[4] = (77.0, 12.0, 0.0)  # 77°12'
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def _plain_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


ANNOTATION = json.dumps({
    "caption": "A woman sits on a rooftop terrace with a golden retriever at sunset.",
    "kind": "photo",
    "people": [{"description": "a woman in a red jacket", "confidence": 0.95}],
    "pets": [{"description": "a golden retriever", "species": "dog", "confidence": 0.9}],
    "objects": ["terrace railing"],
    "environment": "rooftop terrace at sunset",
    "activity": "relaxing with a dog",
    "emotion": "calm",
    "ocr_text": "",
    "place_guess": None,
})
EXTRACTION = json.dumps({"facts": ["Has a golden retriever"]})
DECISION = json.dumps({"event": "ADD", "target_index": None, "text": "Has a golden retriever"})


def test_parse_exif_reads_time_gps_camera():
    info = parse_exif(_jpeg_with_exif())
    assert info.captured_at == datetime(2022, 7, 15, 18, 30, tzinfo=UTC)
    assert info.time_source == "exif"
    assert info.latitude == pytest.approx(28.6, abs=1e-6)
    assert info.longitude == pytest.approx(77.2, abs=1e-6)
    assert info.camera == "TestMake TestCam"
    assert classify_kind(info, "image/jpeg") == "photo"


def test_plain_png_classifies_as_screenshot():
    info = parse_exif(_plain_png())
    assert info.captured_at is None and not info.has_gps and info.camera is None
    assert info.time_source == "upload"
    assert classify_kind(info, "image/png") == "screenshot"


def _make_job(db_session, tmp_path, image_bytes: bytes) -> IngestJob:
    info = parse_exif(image_bytes)
    job = IngestJob(
        user_id="test-user",
        kind=classify_kind(info, "image/jpeg"),
        filename="terrace.jpg",
        content_type="image/jpeg",
        image_path="",
        exif=info.as_context(),
    )
    path = tmp_path / f"{job.id}.jpg"
    path.write_bytes(image_bytes)
    job.image_path = str(path)
    db_session.add(job)
    db_session.flush()
    return job


def test_process_job_creates_episode_once_and_distils(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    llm = FakeLLM([ANNOTATION, EXTRACTION, DECISION])

    processed = process_job(db_session, llm, get_settings(), job_id=job.id)

    assert processed is not None and processed.status == "done"
    assert processed.episode_id is not None
    episode = db_session.get(Episode, processed.episode_id)
    assert episode is not None
    # temporal context came from EXIF, not upload time
    assert episode.occurred_at.replace(tzinfo=UTC) == datetime(2022, 7, 15, 18, 30, tzinfo=UTC)
    assert episode.context["source"] == "image"
    assert episode.context["place"]["latitude"] == pytest.approx(28.6, abs=1e-6)
    types = {e["type"] for e in episode.context["entities"]}
    assert {"person", "pet"} <= types
    # every detected entity awaits a user label — nothing is auto-identified
    assert all(e["label"] is None for e in episode.context["entities"])
    # the caption distilled into a semantic memory with provenance back to the episode
    memories = list(db_session.exec(select(Memory).where(Memory.user_id == "test-user")).all())
    assert len(memories) == 1
    assert memories[0].content == "Has a golden retriever"
    assert memories[0].source_episode_ids == [str(episode.id)]


def test_process_job_geocodes_when_gps_present(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())  # this JPEG carries GPS
    llm = FakeLLM([ANNOTATION, EXTRACTION, DECISION])
    processed = process_job(
        db_session, llm, get_settings(),
        job_id=job.id, geocode=lambda lat, lon: "Jaipur, India",
    )
    assert processed is not None
    episode = db_session.get(Episode, processed.episode_id)
    assert episode is not None
    assert episode.context["place"]["name"] == "Jaipur, India"
    assert episode.context["place"]["latitude"] == pytest.approx(28.6, abs=1e-6)


def test_process_job_no_place_without_geocoder(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    llm = FakeLLM([ANNOTATION, EXTRACTION, DECISION])
    processed = process_job(db_session, llm, get_settings(), job_id=job.id)  # no geocode passed
    assert processed is not None
    episode = db_session.get(Episode, processed.episode_id)
    assert episode is not None
    assert episode.context["place"]["name"] is None  # GPS kept, but no place invented


def test_process_job_is_idempotent(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    llm = FakeLLM([ANNOTATION, EXTRACTION, DECISION])
    process_job(db_session, llm, get_settings(), job_id=job.id)

    # a second run must be a no-op: the status guard stops it before any LLM call
    starved = FakeLLM([])  # would raise IndexError if any call got through
    again = process_job(db_session, starved, get_settings(), job_id=job.id)

    assert again is not None and again.status == "done"
    episodes = list(db_session.exec(select(Episode).where(Episode.user_id == "test-user")).all())
    assert len(episodes) == 1


# the auto-label tests need a pet description that lands in the same fake-embedding slot as
# the seeded "Monty: a small fluffy dog" (keyword "dog"), so candidate matching fires
AUTO_ANNOTATION = json.dumps({
    "caption": "A small fluffy dog nestled in a red blanket.",
    "kind": "photo",
    "people": [],
    "pets": [{"description": "a small fluffy dog", "species": "dog", "confidence": 0.9}],
    "objects": ["red blanket"],
    "environment": "indoors",
    "activity": "resting",
    "emotion": "calm",
    "ocr_text": "",
    "place_guess": None,
})
VERDICT_SAME = json.dumps(
    {"same_subject": True, "confidence": 0.92, "evidence": "same coat and face structure"}
)
VERDICT_UNSURE = json.dumps(
    {"same_subject": True, "confidence": 0.45, "evidence": "similar but partially occluded"}
)
LABEL_DECISION = json.dumps({"event": "NOOP", "target_index": None, "text": ""})


def _seed_known_pet(db_session, tmp_path, user_id: str):
    """Monty already exists: a labeled entity + one USER-confirmed reference photo."""
    from datetime import UTC, datetime

    from app.models import Entity, EpisodeEntity

    ref_episode = Episode(
        user_id=user_id,
        occurred_at=datetime(2023, 5, 22, tzinfo=UTC),
        content="A small, fluffy dog on a blanket.",
        context={"source": "image", "kind": "photo",
                 "entities": [{"type": "pet", "description": "a small fluffy dog", "label": None}]},
        embedding=fake_embedding("a small fluffy dog photo"),
    )
    db_session.add(ref_episode)
    db_session.flush()
    ref_path = tmp_path / "ref.jpg"
    ref_path.write_bytes(_jpeg_with_exif())
    ref_job = IngestJob(
        user_id=user_id, kind="photo", status="done", filename="ref.jpg",
        content_type="image/jpeg", image_path=str(ref_path), exif={},
        episode_id=ref_episode.id,
    )
    monty = Entity(
        user_id=user_id, name="Monty", type="pet", description="a small fluffy dog",
        embedding=fake_embedding("Monty: a small fluffy dog"),
    )
    db_session.add(ref_job)
    db_session.add(monty)
    db_session.flush()
    db_session.add(EpisodeEntity(
        episode_id=ref_episode.id, entity_id=monty.id, entity_index=0, labeled_by="user"
    ))
    db_session.flush()
    return monty


def test_auto_label_recognizes_known_pet(db_session, tmp_path):
    """A new photo of a known pet gets its label applied BY THE MEMORY (visual verify)."""
    from app.models import EpisodeEntity

    user = "test-auto-user"
    monty = _seed_known_pet(db_session, tmp_path, user)
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    job.user_id = user
    db_session.add(job)
    db_session.flush()

    # annotation -> extraction(empty) -> visual verdict -> label fact decision
    llm = FakeLLM([AUTO_ANNOTATION, json.dumps({"facts": []}), VERDICT_SAME, LABEL_DECISION])
    processed = process_job(db_session, llm, get_settings(), job_id=job.id)

    assert processed is not None and processed.status == "done"
    links = list(db_session.exec(
        select(EpisodeEntity).where(EpisodeEntity.episode_id == processed.episode_id)).all())
    assert len(links) == 1
    assert links[0].entity_id == monty.id
    assert links[0].labeled_by == "memory"  # the memory labeled it, visibly


def test_auto_label_declines_below_confidence(db_session, tmp_path):
    """An unsure visual verdict falls back to propose-only — no link is written."""
    from app.models import EpisodeEntity

    user = "test-auto-unsure"
    _seed_known_pet(db_session, tmp_path, user)
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    job.user_id = user
    db_session.add(job)
    db_session.flush()

    llm = FakeLLM([AUTO_ANNOTATION, json.dumps({"facts": []}), VERDICT_UNSURE])
    processed = process_job(db_session, llm, get_settings(), job_id=job.id)

    assert processed is not None and processed.status == "done"
    links = list(db_session.exec(
        select(EpisodeEntity).where(EpisodeEntity.episode_id == processed.episode_id)).all())
    assert links == []


def test_process_job_raises_on_garbage_annotation(db_session, tmp_path):
    job = _make_job(db_session, tmp_path, _jpeg_with_exif())
    llm = FakeLLM(["this is not json at all"])
    with pytest.raises(ValueError):
        process_job(db_session, llm, get_settings(), job_id=job.id)