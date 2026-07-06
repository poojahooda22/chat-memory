"""Reverse-geocoding: format a place, send a User-Agent, cache, and return None on failure —
never a fabricated location. The HTTP call is injected, so the suite never hits the network."""

from app.config import Settings, get_settings
from app.ingest.geocode import _format_place, make_geocoder, reverse_geocode

SETTINGS = get_settings()

NOMINATIM = {
    "display_name": "Hawa Mahal, Jaipur, Rajasthan, India",
    "address": {"city": "Jaipur", "state": "Rajasthan", "country": "India"},
}


def test_format_prefers_city_country():
    assert _format_place(NOMINATIM) == "Jaipur, India"


def test_format_none_on_empty():
    assert _format_place(None) is None
    assert _format_place({}) is None


def test_reverse_geocode_uses_injected_fetch_and_sends_user_agent():
    seen: dict[str, str] = {}

    def fake_get(url, headers):
        seen["url"] = url
        seen["ua"] = headers.get("User-Agent", "")
        return NOMINATIM

    name = reverse_geocode(26.925, 75.826, settings=SETTINGS, http_get=fake_get)
    assert name == "Jaipur, India"
    assert "lat=26.925" in seen["url"] and "lon=75.826" in seen["url"]
    assert seen["ua"]  # Nominatim requires a User-Agent


def test_reverse_geocode_none_on_failure():
    def boom(url, headers):
        raise RuntimeError("network down")

    assert reverse_geocode(1.5, 2.5, settings=SETTINGS, http_get=boom) is None


def test_make_geocoder_disabled_returns_none():
    assert make_geocoder(Settings(geocode_enabled=False)) is None