"""Validation and normalization for external job links."""

from urllib.parse import urlparse


def normalize_job_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip().strip("<>")
    if url.lower().startswith("www."):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return url
