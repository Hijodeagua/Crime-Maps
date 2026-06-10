"""
Shared HTTP session for all outbound data fetches.

Municipal ArcGIS servers (and some WAFs in front of them) commonly reject
requests with non-browser User-Agents or missing Accept headers — returning
403 even though the endpoint is public. Every fetch in this project goes
through make_session() / get() so the headers are consistent and fixable in
one place.
"""

import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


_shared = None


def get(url: str, **kwargs) -> requests.Response:
    """Module-level GET using a shared browser-headed session."""
    global _shared
    if _shared is None:
        _shared = make_session()
    return _shared.get(url, **kwargs)
