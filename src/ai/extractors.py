"""Text extraction for supported document formats.

Each extractor returns an :class:`ExtractedDocument` carrying the plain text
plus optional page-split text (PDF) so page numbers survive chunking. File
content is always treated as *data*, never as instructions.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ai.config import SUPPORTED_EXTENSIONS
from src.ai.errors import EmptyDocumentError, ExtractionError, UnsupportedFormatError
from src.utils.logger import get_logger

log = get_logger(__name__)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class ExtractedDocument:
    """Normalised text payload produced by an extractor."""

    text: str
    pages: list[str] = field(default_factory=list)
    title: str | None = None
    meta: dict = field(default_factory=dict)
    extractor: str = ""


def normalize_text(text: str) -> str:
    """Strip control characters, normalise newlines and collapse blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def supported(filename: str) -> bool:
    return extension_of(filename) in SUPPORTED_EXTENSIONS


def extract_bytes(data: bytes, filename: str) -> ExtractedDocument:
    """Dispatch to the extractor matching ``filename``'s extension."""
    ext = extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"unsupported file type '{ext or filename}' - supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not data:
        raise EmptyDocumentError(f"'{filename}' is empty")
    kind = SUPPORTED_EXTENSIONS[ext]
    try:
        if kind == "pdf":
            doc = _extract_pdf(data, filename)
        elif kind == "json":
            doc = _extract_json(data, filename)
        elif kind == "csv":
            doc = _extract_csv(data, filename)
        elif kind == "html":
            doc = _extract_html(data, filename)
        else:
            doc = _extract_text(data, filename)
    except (UnsupportedFormatError, EmptyDocumentError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"could not read '{filename}': {exc}") from exc
    if not doc.text.strip():
        raise EmptyDocumentError(f"'{filename}' contains no extractable text")
    return doc


def extract_path(path: Path) -> ExtractedDocument:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"could not read {path}: {exc}") from exc
    return extract_bytes(data, path.name)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("file is not decodable as text")


def _extract_text(data: bytes, filename: str) -> ExtractedDocument:
    text = normalize_text(_decode(data))
    title = Path(filename).stem.replace("_", " ").replace("-", " ") or None
    return ExtractedDocument(text=text, title=title, extractor="text")


def _extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ExtractionError("pypdf is not installed (pip install pypdf)") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable PDF: {exc}") from exc
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(normalize_text(page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001 - one broken page must not kill the file
            log.warning("PDF page extraction failed in %s: %s", filename, exc)
            pages.append("")
    text = normalize_text("\n\n".join(pages))
    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
    except Exception:  # noqa: BLE001 - metadata is best effort
        title = None
    return ExtractedDocument(
        text=text,
        pages=[p for p in pages],
        title=title or Path(filename).stem,
        extractor="pypdf",
        meta={"page_count": len(pages)},
    )


def _extract_html(data: bytes, filename: str) -> ExtractedDocument:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("beautifulsoup4 is not installed") from exc
    soup = BeautifulSoup(_decode(data), "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    text = normalize_text(soup.get_text(separator="\n"))
    return ExtractedDocument(text=text, title=title, extractor="html")


def _flatten_json(value, prefix: str = "", depth: int = 0) -> list[str]:
    lines: list[str] = []
    if depth > 12:
        return lines
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(item, path, depth + 1))
    elif isinstance(value, list):
        for i, item in enumerate(value[:500]):
            lines.extend(_flatten_json(item, f"{prefix}[{i}]", depth + 1))
    elif value is None:
        lines.append(f"{prefix}: null")
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def _extract_json(data: bytes, filename: str) -> ExtractedDocument:
    try:
        payload = json.loads(_decode(data))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExtractionError(f"invalid JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        text = normalize_text(payload["text"])
        title = str(payload.get("title") or Path(filename).stem)
    else:
        text = normalize_text("\n".join(_flatten_json(payload)))
        title = Path(filename).stem
    return ExtractedDocument(text=text, title=title, extractor="json")


def _extract_csv(data: bytes, filename: str) -> ExtractedDocument:
    raw = _decode(data)
    try:
        rows = list(csv.reader(io.StringIO(raw)))
    except csv.Error as exc:
        raise ExtractionError(f"invalid CSV: {exc}") from exc
    if not rows:
        raise EmptyDocumentError(f"'{filename}' has no rows")
    header, *body = rows
    lines = [" | ".join(str(cell).strip() for cell in header)]
    for row in body[:5000]:
        if not any(cell.strip() for cell in row):
            continue
        lines.append(" | ".join(str(cell).strip() for cell in row))
    return ExtractedDocument(
        text=normalize_text("\n".join(lines)),
        title=Path(filename).stem,
        extractor="csv",
        meta={"rows": len(body)},
    )
