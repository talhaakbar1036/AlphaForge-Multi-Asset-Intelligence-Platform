"""Security helpers for the copilot: uploads, untrusted text, rate limits.

Three separate concerns live here on purpose:

* **file safety** - path traversal, size caps, extension allow-lists;
* **untrusted retrieved text** - everything that comes out of a document is
  wrapped and de-weaponised before it reaches a prompt;
* **rate limiting** - a small in-process sliding window so a chatty client
  cannot pin the CPU with embedding + LLM calls.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from pathlib import Path

from src.ai.errors import (
    InvalidFilenameError,
    RateLimitedError,
    UnauthorizedError,
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

# Patterns that *claim* to override the system prompt inside retrieved text.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions",
        r"disregard\s+(?:the\s+)?(?:system|previous|above)\s+prompt",
        r"you\s+are\s+now\s+(?:a|an|the)\s+",
        r"system\s*prompt\s*:",
        r"do\s+not\s+tell\s+the\s+user",
        r"exfiltrat|api\s*key\s*[:=]|password\s*[:=]|secret\s*[:=]",
        r"<\s*/?\s*(?:system|assistant)\s*>",
    )
]

MAX_FLAGGED_PATTERNS = 8


def sanitize_filename(filename: str) -> str:
    """Return a safe basename; reject traversal and control characters."""
    raw = (filename or "").strip()
    if not raw:
        raise InvalidFilenameError("filename is empty")
    if "\x00" in raw:
        raise InvalidFilenameError("filename contains a null byte")
    cleaned = _CONTROL_RE.sub("", raw).replace("\\", "/")
    name = Path(cleaned).name
    if not name or name in {".", ".."}:
        raise InvalidFilenameError(f"invalid filename: {filename!r}")
    if ".." in name:
        raise InvalidFilenameError(f"path traversal rejected: {filename!r}")
    if len(name) > 200:
        raise InvalidFilenameError("filename is too long (max 200 characters)")
    return name


def safe_storage_name(document_id: str, filename: str) -> str:
    """Deterministic, traversal-free name used when storing the original file."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)[:120]
    return f"{document_id}__{stem}"


def validate_upload(filename: str, size: int, max_bytes: int) -> str:
    name = sanitize_filename(filename)
    if size <= 0:
        raise InvalidFilenameError(f"'{name}' is empty")
    if size > max_bytes:
        from src.ai.errors import DocumentTooLargeError

        raise DocumentTooLargeError(
            f"'{name}' is {size / (1024 * 1024):.1f} MB which exceeds the "
            f"{max_bytes // (1024 * 1024)} MB upload limit"
        )
    return name


def scrub_control_chars(text: str, limit: int = 200_000) -> str:
    """Remove zero-width/control characters that could hide instructions."""
    cleaned = _ZERO_WIDTH_RE.sub("", _CONTROL_RE.sub("", text or ""))
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + "\n[truncated]"
    return cleaned


def detect_injection(text: str) -> list[str]:
    """Return the suspicious patterns found (report-only, never blocking)."""
    found: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            found.append(match.group(0)[:80])
            if len(found) >= MAX_FLAGGED_PATTERNS:
                break
    return found


def wrap_untrusted(text: str, *, label: str = "document") -> str:
    """Fence retrieved text and state that it is data, not instructions."""
    cleaned = scrub_control_chars(text)
    return (
        f"<<<UNTRUSTED_{label.upper()}>>>\n"
        f"{cleaned}\n"
        f"<<<END_UNTRUSTED_{label.upper()}>>>"
    )


def check_admin_token(presented: str | None, expected: str) -> None:
    """Constant-ish comparison of the admin bearer token.

    When ``expected`` is empty the deployment runs in local-dev mode and no
    token is required (health reporting makes that state visible).
    """
    if not expected:
        return
    if not presented:
        raise UnauthorizedError("admin token required (Authorization: Bearer <token>)")
    import hmac

    if not hmac.compare_digest(presented.strip(), expected):
        raise UnauthorizedError("invalid admin token")


class SlidingWindowRateLimiter:
    """Per-key request counter over a rolling one-minute window."""

    def __init__(self, limit_per_minute: int):
        self.limit = int(limit_per_minute)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def check(self, key: str = "default") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self.limit:
                raise RateLimitedError(
                    f"rate limit of {self.limit} requests/minute exceeded - retry shortly"
                )
            window.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
