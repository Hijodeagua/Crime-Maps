"""
Shared HTTP session for all outbound data fetches.

Municipal ArcGIS servers (and some WAFs in front of them) commonly reject
requests with non-browser User-Agents or missing Accept headers — returning
403 even though the endpoint is public. Every fetch in this project goes
through make_session() / get() so the headers are consistent and fixable in
one place.

Sessions built here also retry transient failures (connection resets, 429
rate limits, 5xx responses) with exponential backoff via urllib3's Retry, so
a single flaky response doesn't fail an entire refresh or page load. Every
request gets DEFAULT_TIMEOUT unless the caller passes an explicit timeout,
so no fetch can hang forever.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Applied to every request that doesn't pass an explicit timeout.
DEFAULT_TIMEOUT = 30  # seconds

RETRY_TOTAL = 3
RETRY_BACKOFF_FACTOR = 1.0  # sleeps ~1s, 2s, 4s between attempts
RETRY_STATUSES = (429, 500, 502, 503, 504)


class _DefaultTimeoutSession(requests.Session):
    """requests.Session that applies DEFAULT_TIMEOUT when none is given."""

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        return super().request(method, url, **kwargs)


def make_session(
    retries: int = RETRY_TOTAL,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
) -> requests.Session:
    """Browser-headed session that retries idempotent GETs on transient errors."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff_factor,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = _DefaultTimeoutSession()
    session.headers.update(DEFAULT_HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_shared = None


def get(url: str, **kwargs) -> requests.Response:
    """Module-level GET using a shared browser-headed, retrying session."""
    global _shared
    if _shared is None:
        _shared = make_session()
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return _shared.get(url, **kwargs)
