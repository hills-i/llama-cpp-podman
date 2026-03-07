"""Shared HTTP session factory with connection pooling and retry logic."""

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_http_session(
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    retries: int = 3,
    backoff_factor: float = 1,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> Session:
    """Create a requests.Session with connection pooling and automatic retries."""
    session = Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
