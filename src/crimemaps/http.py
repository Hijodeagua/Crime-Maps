"""
Shared HTTP session with retry/backoff for upstream APIs.

All outbound requests (ArcGIS, Census ACS, TIGERweb) go through a session
built here so transient 5xx/429 responses and connection resets are retried
with exponential backoff instead of failing the whole refresh.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_TOTAL = 3
_BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s
_RETRY_STATUSES = (429, 500, 502, 503, 504)


def retrying_session(
    total: int = _RETRY_TOTAL,
    backoff_factor: float = _BACKOFF_FACTOR,
) -> requests.Session:
    """Return a requests.Session that retries idempotent GETs on transient errors."""
    retry = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
