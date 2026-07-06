"""Reverse-geocoding: turn a photo's lat/lon into a human place name ("Jaipur, India").

An outbound fetch, so it lives in the WORKER (never the request path), and it is cached by
rounded coordinates so the same spot is never looked up twice — respecting Nominatim's usage
policy (single occasional lookups only; bulk needs a self-hosted instance or a paid geocoder).

It returns None on ANY failure or ambiguity — a photo with no resolvable place stays
"location unknown", never a fabricated one (the project's never-invent rule applied to place).
"""

import json
import urllib.parse
import urllib.request
from collections.abc import Callable

from app.config import Settings

# rounded (lat, lon) -> place name | None. Keyed at ~100m so one place is looked up once; the
# set of distinct places for a single user is inherently small, so this stays bounded.
_CACHE: dict[tuple[float, float], str | None] = {}
_CACHE_CAP = 5000


def _default_http_get(url: str, headers: dict[str, str], timeout: float = 8.0) -> dict | None:
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - https URL from settings
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _format_place(data: dict | None) -> str | None:
    """A concise 'City, Country' from a Nominatim response — else its display name, else None."""
    if not data:
        return None
    address = data.get("address") or {}
    locality = next(
        (address[k] for k in ("city", "town", "village", "suburb", "county", "state_district")
         if address.get(k)),
        None,
    )
    region = address.get("state")
    country = address.get("country")
    if locality and country:
        return f"{locality}, {country}"
    if region and country:
        return f"{region}, {country}"
    return data.get("display_name") or country


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    settings: Settings,
    http_get: Callable[[str, dict[str, str]], dict | None] = _default_http_get,
) -> str | None:
    key = (round(lat, 3), round(lon, 3))
    if key in _CACHE:
        return _CACHE[key]
    query = urllib.parse.urlencode({
        "lat": f"{lat}", "lon": f"{lon}", "format": "json",
        "zoom": settings.geocode_zoom, "accept-language": "en",
    })
    try:
        data = http_get(f"{settings.geocode_url}?{query}", {"User-Agent": settings.geocode_user_agent})
        name = _format_place(data)
    except Exception:
        name = None  # network/parse failure → location unknown, never invented
    if len(_CACHE) < _CACHE_CAP:
        _CACHE[key] = name
    return name


def make_geocoder(settings: Settings) -> Callable[[float, float], str | None] | None:
    """The worker's geocoder, or None when disabled — so tests and offline runs skip the fetch."""
    if not settings.geocode_enabled:
        return None
    return lambda lat, lon: reverse_geocode(lat, lon, settings=settings)