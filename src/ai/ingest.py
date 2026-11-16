"""Document ingestion: bytes -> text -> chunks -> vectors -> registry.

Invariants enforced here (see the security/architecture notes):

* an upload is validated *before* any bytes touch the filesystem;
* duplicate content is detected by SHA-256 and never indexed twice;
* an existing document id is only replaced when the caller asked for it and
  the response reports ``status="updated"`` (never a silent swap);
* the vector index and the registry are updated together, and deletion
  removes chunks from both.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.ai.chunking import Chunk, chunk_document
from src.ai.config import AIConfig, SUPPORTED_EXTENSIONS
from src.ai.errors import (
    DocumentError,
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFormatError,
)
from src.ai.extractors import extract_bytes, supported
from src.ai.registry import DocumentRecord, DocumentRegistry, validate_document_id
from src.ai.security import sanitize_filename, safe_storage_name, validate_upload
from src.ai.vectorstore import (
    KEY_CHUNK_ID,
    KEY_DOC_ID,
    KEY_FILENAME,
    KEY_PAGE,
    KEY_PUB_TS,
    KEY_TICKER,
    KEY_TITLE,
    KEY_DOC_TYPE,
    KEY_PUB_DATE,
    KEY_SOURCE_URL,
    KEY_VISIBILITY,
    VectorStore,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_DOCUMENT_CHARS = 2_000_000

_DOC_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # lookarounds instead of \b so names like "NVDA_10-K_2025.pdf" match
    ("10-K", re.compile(r"(?<![A-Za-z0-9])10[-_]?k(?![A-Za-z0-9])|annual[-_ ]report", re.I)),
    ("10-Q", re.compile(r"(?<![A-Za-z0-9])10[-_]?q(?![A-Za-z0-9])|quarterly[-_ ]report", re.I)),
    ("8-K", re.compile(r"(?<![A-Za-z0-9])8[-_]?k(?![A-Za-z0-9])", re.I)),
    ("earnings", re.compile(r"earnings|results[-_ ]release|press[-_ ]release", re.I)),
    ("filing", re.compile(r"\bs-1\b|\b424b\b|prospectus|filing", re.I)),
    ("model-evaluation", re.compile(r"evaluation|model[-_ ]?card|validation", re.I)),
    ("backtest", re.compile(r"backtest", re.I)),
    ("risk-report", re.compile(r"\brisk[-_ ]report\b|\brisk[-_ ]profile\b", re.I)),
    ("eda-report", re.compile(r"\beda\b|exploratory", re.I)),
    ("news", re.compile(r"\bnews\b|headline", re.I)),
    ("documentation", re.compile(r"readme|docs?[-_]|architecture|guide", re.I)),
]


def guess_doc_type(filename: str) -> str | None:
    for label, pattern in _DOC_TYPE_PATTERNS:
        if pattern.search(filename):
            return label
    return None


def make_document_id(filename: str, ticker: str | None = None) -> str:
    """Stable id derived from (ticker, filename) so re-uploads update in place."""
    key = f"{(ticker or '').upper().strip()}:{sanitize_filename(filename).lower()}"
    return "doc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _parse_date(value: str | None) -> tuple[str | None, float | None]:
    """Return (ISO date string, unix ts) for metadata + range filtering."""
    if not value:
        return None, None
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%d %b %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return dt.strftime("%Y-%m-%d"), dt.timestamp()
    return None, None


@dataclass
class IngestResult:
    """Outcome of one ingestion (or the refusal of one)."""

    document_id: str
    filename: str
    status: str                      # indexed | updated | duplicate
    chunks: int = 0
    content_hash: str = ""
    ticker: str | None = None
    doc_type: str | None = None
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    document: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "status": self.status,
            "chunks": self.chunks,
            "content_hash": self.content_hash,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "title": self.title,
            "warnings": self.warnings,
            "document": self.document,
        }


class DocumentIngestor:
    """Owns the ingest/evict logic for one configured store + registry."""

    def __init__(
        self,
        cfg: AIConfig,
        store: VectorStore,
        registry: DocumentRegistry,
        embedder,
    ):
        self.cfg = cfg
        self.store = store
        self.registry = registry
        self.embedder = embedder

    # ------------------------------------------------------------- ingest --
    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        ticker: str | None = None,
        doc_type: str | None = None,
        title: str | None = None,
        publication_date: str | None = None,
        reporting_period: str | None = None,
        source_url: str | None = None,
        visibility: str = "public",
        replace: bool = True,
    ) -> IngestResult:
        name = validate_upload(filename, len(data), self.cfg.max_upload_bytes)
        if not supported(name):
            raise UnsupportedFormatError(
                f"unsupported file type '{Path(name).suffix}' - supported: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if visibility not in {"public", "private"}:
            raise DocumentError("visibility must be 'public' or 'private'")

        content_hash = hashlib.sha256(data).hexdigest()
        ticker = (ticker or "").strip().upper() or None
        document_id = make_document_id(name, ticker)
        warnings: list[str] = []

        # --- duplicate detection (by content, across ids) ------------------
        same_content = self.registry.get_by_hash(content_hash)
        if same_content is not None and same_content.document_id != document_id:
            return IngestResult(
                document_id=same_content.document_id,
                filename=same_content.filename,
                status="duplicate",
                chunks=same_content.chunk_count,
                content_hash=content_hash,
                ticker=same_content.ticker,
                doc_type=same_content.doc_type,
                title=same_content.title,
                warnings=[
                    f"identical content is already indexed as "
                    f"'{same_content.document_id}' ({same_content.filename})"
                ],
                document=same_content.to_dict(),
            )

        existing = self.registry.get(document_id)
        if existing is not None:
            if existing.content_hash == content_hash:
                return IngestResult(
                    document_id=document_id,
                    filename=existing.filename,
                    status="duplicate",
                    chunks=existing.chunk_count,
                    content_hash=content_hash,
                    ticker=existing.ticker,
                    doc_type=existing.doc_type,
                    title=existing.title,
                    warnings=["content unchanged - index left untouched"],
                    document=existing.to_dict(),
                )
            if not replace:
                raise DocumentError(
                    f"document '{document_id}' already exists with different content "
                    "(pass replace=true to overwrite it)"
                )

        # --- extract + chunk + embed (no side effects yet) ------------------
        doc = extract_bytes(data, name)
        if len(doc.text) > MAX_DOCUMENT_CHARS:
            doc.text = doc.text[:MAX_DOCUMENT_CHARS]
            warnings.append(f"document truncated to {MAX_DOCUMENT_CHARS} characters")
        chunks = chunk_document(doc, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            raise EmptyDocumentError(f"'{name}' produced no chunkable text")

        texts = [c.text for c in chunks]
        try:
            vectors = self.embedder.encode(texts)
        except Exception as exc:  # noqa: BLE001
            raise DocumentError(f"embedding failed for '{name}': {exc}") from exc
        if vectors.shape[0] != len(chunks):
            raise DocumentError("embedding backend returned the wrong vector count")

        pub_date, pub_ts = _parse_date(publication_date)
        final_title = (title or doc.title or Path(name).stem).strip()
        final_doc_type = (doc_type or guess_doc_type(name) or "document").strip()

        base_meta = {
            KEY_DOC_ID: document_id,
            KEY_FILENAME: name,
            KEY_TITLE: final_title,
            KEY_DOC_TYPE: final_doc_type,
            KEY_VISIBILITY: visibility,
            "embedding_backend": self.embedder.identity,
            "ingested_ts": datetime.now(timezone.utc).timestamp(),
            "content_hash": content_hash,
        }
        if ticker:
            base_meta[KEY_TICKER] = ticker
        if pub_date:
            base_meta[KEY_PUB_DATE] = pub_date
            base_meta[KEY_PUB_TS] = pub_ts
        if source_url:
            base_meta[KEY_SOURCE_URL] = str(source_url)[:500]
        if reporting_period:
            base_meta["reporting_period"] = str(reporting_period)[:40]

        ids, metas = [], []
        for chunk in chunks:
            chunk_id = Chunk.make_id(document_id, chunk.index)
            ids.append(chunk_id)
            meta = dict(base_meta)
            meta[KEY_CHUNK_ID] = chunk_id
            meta["chunk_index"] = chunk.index
            meta["start_offset"] = chunk.start
            if chunk.page_number is not None:
                meta[KEY_PAGE] = int(chunk.page_number)
            metas.append(meta)

        # --- commit: vectors first, then stale-chunk prune, then registry ----
        self.store.add(ids, vectors, texts, metas)
        if existing is not None and existing.chunk_count > len(ids):
            stale = [Chunk.make_id(document_id, i) for i in range(len(ids), existing.chunk_count)]
            self.store.delete(ids=stale)
        if existing is not None:
            warnings.append(
                f"replaced previous version (hash {existing.content_hash[:12]}…, "
                f"{existing.chunk_count} chunks)"
            )

        stored_path = self._store_original(data, document_id, name)
        record = DocumentRecord(
            document_id=document_id,
            filename=name,
            content_hash=content_hash,
            title=final_title,
            ticker=ticker,
            doc_type=final_doc_type,
            publication_date=pub_date,
            reporting_period=reporting_period,
            source_url=source_url,
            visibility=visibility,
            byte_size=len(data),
            chunk_count=len(ids),
            status="indexed",
            stored_path=str(stored_path) if stored_path else None,
            embedding_backend=self.embedder.identity,
            extra={"extractor": doc.extractor, "warnings": warnings},
        )
        self.registry.upsert(record)
        status = "updated" if existing is not None else "indexed"
        log.info(
            "%s document %s (%s, %d chunks, ticker=%s)",
            status,
            document_id,
            final_doc_type,
            len(ids),
            ticker or "-",
        )
        return IngestResult(
            document_id=document_id,
            filename=name,
            status=status,
            chunks=len(ids),
            content_hash=content_hash,
            ticker=ticker,
            doc_type=final_doc_type,
            title=final_title,
            warnings=warnings,
            document=record.to_dict(),
        )

    def ingest_path(self, path: Path, **meta: Any) -> IngestResult:
        path = Path(path)
        data = path.read_bytes()
        return self.ingest_bytes(data, path.name, **meta)

    def ingest_text(
        self,
        text: str,
        filename: str,
        **meta: Any,
    ) -> IngestResult:
        """Convenience wrapper used by tests and the CLI."""
        if not text or not text.strip():
            raise EmptyDocumentError(f"'{filename}' contains no text")
        return self.ingest_bytes(text.encode("utf-8"), filename, **meta)

    def ingest_paths(self, paths: Iterable[Path], **meta: Any) -> list[IngestResult]:
        results: list[IngestResult] = []
        for path in sorted(Path(p) for p in paths):
            try:
                results.append(self.ingest_path(path, **meta))
            except (UnsupportedFormatError, ExtractionError, EmptyDocumentError) as exc:
                log.warning("skipping %s: %s", path.name, exc)
        return results

    # ------------------------------------------------------------- evict --
    def delete(self, document_id: str) -> bool:
        """Remove a document from *both* the vector index and the registry."""
        doc_id = validate_document_id(document_id)
        record = self.registry.get(doc_id)
        if record is None:
            return False
        removed = self.store.delete(where={KEY_DOC_ID: doc_id})
        self.registry.delete(doc_id)
        self._remove_original(record)
        log.info("deleted document %s (%d chunks removed)", doc_id, removed)
        return True

    # ----------------------------------------------------------- helpers --
    def _store_original(self, data: bytes, document_id: str, filename: str) -> Path | None:
        try:
            target_dir = self.cfg.documents_dir / document_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / safe_storage_name(document_id, filename)
            target.write_bytes(data)
            return target
        except OSError as exc:  # storage is best effort - index already committed
            log.warning("could not store original file for %s: %s", document_id, exc)
            return None

    def _remove_original(self, record: DocumentRecord) -> None:
        if not record.stored_path:
            return
        try:
            path = Path(record.stored_path)
            if path.is_file() and self.cfg.documents_dir in path.parents:
                path.unlink()
                directory = path.parent
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
        except OSError as exc:
            log.warning("could not remove stored file for %s: %s", record.document_id, exc)

    # ----------------------------------------------------------- queries --
    def list_documents(
        self,
        ticker: str | None = None,
        doc_type: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.registry.list(ticker, doc_type, visibility)]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        record = self.registry.get(validate_document_id(document_id))
        return record.to_dict() if record else None


def content_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
