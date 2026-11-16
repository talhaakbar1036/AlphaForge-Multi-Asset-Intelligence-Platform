"""Vector index abstraction.

``ChromaVectorStore`` (default) persists to ``data/vectorstore`` via ChromaDB.
``InMemoryVectorStore`` is a pure-NumPy fallback so the copilot still works
when chromadb is not installed - degraded, but never dead.

Both stores keep the full chunk text and its metadata next to the vector,
which is what makes source citations possible without a second database.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.ai.errors import VectorStoreError
from src.utils.logger import get_logger

log = get_logger(__name__)

# Metadata keys shared by every chunk
KEY_DOC_ID = "document_id"
KEY_CHUNK_ID = "chunk_id"
KEY_VISIBILITY = "visibility"
KEY_TICKER = "ticker"
KEY_DOC_TYPE = "doc_type"
KEY_TITLE = "title"
KEY_FILENAME = "filename"
KEY_PAGE = "page_number"
KEY_PUB_TS = "publication_ts"


@dataclass
class SearchHit:
    """One retrieved chunk with its provenance."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0

    @property
    def document_id(self) -> str:
        return str(self.metadata.get(KEY_DOC_ID, ""))

    @property
    def page_number(self) -> int | None:
        page = self.metadata.get(KEY_PAGE)
        try:
            return int(page) if page is not None else None
        except (TypeError, ValueError):
            return None

    def as_source(self) -> dict[str, Any]:
        """Public citation payload (never includes raw chunk text)."""
        return {
            "document_id": self.document_id,
            "chunk_id": str(self.metadata.get(KEY_CHUNK_ID, self.id)),
            "title": self.metadata.get(KEY_TITLE),
            "filename": self.metadata.get(KEY_FILENAME),
            "ticker": self.metadata.get(KEY_TICKER),
            "doc_type": self.metadata.get(KEY_DOC_TYPE),
            "publication_date": self.metadata.get(KEY_PUB_DATE),
            "source_url": self.metadata.get(KEY_SOURCE_URL),
            "page_number": self.page_number,
            "score": round(float(self.score), 4),
            "retrieval_rank": self.rank,
        }


KEY_PUB_DATE = "publication_date"
KEY_SOURCE_URL = "source_url"


def _cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(vector) + 1e-12)
    return (matrix @ vector) / np.where(denom == 0, 1.0, denom)


class VectorStore(ABC):
    """Common interface for the vector index backends."""

    backend: str = "base"

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def document_ids(self) -> set[str]: ...


# --------------------------------------------------------------------- chroma
class ChromaVectorStore(VectorStore):
    """ChromaDB persistent index."""

    backend = "chroma"

    def __init__(self, path: Path, collection: str, embedding_identity: str):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - env dependent
            raise VectorStoreError(
                "chromadb is not installed; pip install chromadb "
                "or set ALPHAFORGE_VECTOR_BACKEND=memory"
            ) from exc
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine", "embedding_identity": embedding_identity},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not open vector index at {path}: {exc}") from exc
        stored = self._collection.metadata or {}
        previous = stored.get("embedding_identity")
        if previous and previous != embedding_identity:
            raise VectorStoreError(
                f"vector index at {path} was built with embeddings "
                f"'{previous}' but the active backend is '{embedding_identity}'. "
                "Re-ingest the documents or restore the previous "
                "ALPHAFORGE_EMBEDDING_* settings."
            )
        self._identity = embedding_identity

    def add(self, ids, embeddings, texts, metadatas) -> None:
        if not ids:
            return
        try:
            self._collection.upsert(
                ids=list(ids),
                embeddings=[np.asarray(v, dtype=np.float32).tolist() for v in embeddings],
                documents=[str(t) for t in texts],
                metadatas=[_clean_meta(m) for m in metadatas],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"vector index write failed: {exc}") from exc

    def query(self, vector, k, where=None) -> list[SearchHit]:
        if self.count() == 0:
            return []
        try:
            raw = self._collection.query(
                query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                n_results=max(int(k), 1),
                where=_chroma_where(where),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"vector search failed: {exc}") from exc
        hits: list[SearchHit] = []
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        for i, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists, strict=False)):
            distance = float(dist)
            hits.append(
                SearchHit(
                    id=str(cid),
                    text=text or "",
                    metadata=dict(meta or {}),
                    score=max(0.0, 1.0 - distance),
                    rank=i + 1,
                )
            )
        return hits

    def delete(self, ids=None, where=None) -> int:
        try:
            if ids:
                targets = [str(i) for i in ids]
            elif where:
                payload = self._collection.get(where=_chroma_where(where), include=["metadatas"])
                targets = [str(i) for i in (payload.get("ids") or [])]
            else:
                return 0
            if not targets:
                return 0
            self._collection.delete(ids=targets)
            return len(targets)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"vector index delete failed: {exc}") from exc

    def count(self) -> int:
        return int(self._collection.count())

    def document_ids(self) -> set[str]:
        try:
            payload = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not list indexed documents: {exc}") from exc
        return {
            str(m.get(KEY_DOC_ID))
            for m in (payload.get("metadatas") or [])
            if m and m.get(KEY_DOC_ID)
        }


# ------------------------------------------------------------------ in-memory
class InMemoryVectorStore(VectorStore):
    """NumPy brute-force index (chromadb fallback / unit tests)."""

    backend = "memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, ids, embeddings, texts, metadatas) -> None:
        if not ids:
            return
        if not (len(ids) == len(texts) == len(metadatas)):
            raise VectorStoreError("ids, texts and metadatas must have the same length")
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise VectorStoreError("embeddings must be a 2-D array")
        for cid in ids:
            if cid in self._ids:
                self.delete(ids=[cid])
        if self._matrix.size and self._matrix.shape[1] != emb.shape[1]:
            raise VectorStoreError(
                f"dimension mismatch: index has {self._matrix.shape[1]}-d vectors, "
                f"new vectors are {emb.shape[1]}-d"
            )
        self._ids.extend(str(i) for i in ids)
        self._texts.extend(str(t) for t in texts)
        self._metas.extend(dict(m) for m in metadatas)
        self._matrix = np.vstack([self._matrix, emb]) if self._matrix.size else emb

    def query(self, vector, k, where=None) -> list[SearchHit]:
        if not self._ids:
            return []
        scores = _cosine_scores(self._matrix, np.asarray(vector, dtype=np.float32))
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            if len(hits) >= k:
                break
            meta = self._metas[int(idx)]
            if where and not _match(meta, where):
                continue
            hits.append(
                SearchHit(
                    id=self._ids[int(idx)],
                    text=self._texts[int(idx)],
                    metadata=meta,
                    score=float(scores[int(idx)]),
                    rank=len(hits) + 1,
                )
            )
        return hits

    def delete(self, ids=None, where=None) -> int:
        keep_ids = set(ids or [])
        removed = 0
        new_ids, new_texts, new_metas, keep_rows = [], [], [], []
        for i, (cid, text, meta) in enumerate(zip(self._ids, self._texts, self._metas, strict=True)):
            drop = (ids and cid in keep_ids) or (where and _match(meta, where))
            if drop:
                removed += 1
                continue
            new_ids.append(cid)
            new_texts.append(text)
            new_metas.append(meta)
            keep_rows.append(i)
        if removed:
            self._ids, self._texts, self._metas = new_ids, new_texts, new_metas
            self._matrix = self._matrix[keep_rows] if keep_rows else np.zeros((0, 0), dtype=np.float32)
        return removed

    def count(self) -> int:
        return len(self._ids)

    def document_ids(self) -> set[str]:
        return {str(m.get(KEY_DOC_ID)) for m in self._metas if m.get(KEY_DOC_ID)}


# -------------------------------------------------------------------- helpers
def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            clean[key] = value.item()
        else:
            clean[key] = str(value)
    return clean


def _chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate our generic filter DSL into a Chroma ``where`` clause.

    Chroma requires a *single* top-level operator per ``where`` dict: more than
    one field constraint must be wrapped in ``{"$and": [...]}``. Getting this
    wrong raises ``ValueError: Expected where to have exactly one operator``.
    """
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    if where.get("any_of"):
        for sub in where["any_of"]:
            single = _chroma_where(sub)
            if single:
                clauses.append(single)
        if not clauses:
            return None
        return {"$or": clauses} if len(clauses) > 1 else clauses[0]
    for key in (KEY_VISIBILITY, KEY_TICKER, KEY_DOC_TYPE, KEY_DOC_ID):
        if key in where:
            value = where[key]
            clauses.append(
                {key: {"$in": list(value)} if isinstance(value, list) else {"$eq": value}}
            )
    for key in (KEY_PUB_TS, KEY_PAGE, "ingested_ts"):
        rng = where.get(f"{key}_range")
        if rng:
            inner: dict[str, Any] = {}
            if rng.get("gte") is not None:
                inner["$gte"] = float(rng["gte"])
            if rng.get("lte") is not None:
                inner["$lte"] = float(rng["lte"])
            if inner:
                clauses.append({key: inner})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _match(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """In-memory equivalent of :func:`_chroma_where`."""
    if where.get("any_of"):
        return any(_match(meta, sub) for sub in where["any_of"])
    for key in (KEY_VISIBILITY, KEY_TICKER, KEY_DOC_TYPE, KEY_DOC_ID):
        if key in where:
            value = where[key]
            options = list(value) if isinstance(value, list) else [value]
            if meta.get(key) not in options:
                return False
    for key in (KEY_PUB_TS, KEY_PAGE, "ingested_ts"):
        rng = where.get(f"{key}_range")
        if not rng:
            continue
        raw = meta.get(key)
        if raw is None:
            return False
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return False
        if rng.get("gte") is not None and number < float(rng["gte"]):
            return False
        if rng.get("lte") is not None and number > float(rng["lte"]):
            return False
    return True


def build_vector_store(path: Path, collection: str, embedding_identity: str) -> VectorStore:
    """Chroma when available (and requested), NumPy fallback otherwise."""
    import os

    requested = os.getenv("ALPHAFORGE_VECTOR_BACKEND", "auto").strip().lower() or "auto"
    if requested == "memory":
        return InMemoryVectorStore()
    try:
        return ChromaVectorStore(path, collection, embedding_identity)
    except VectorStoreError:
        if requested == "chroma":
            raise
        log.warning(
            "chromadb unavailable - using the in-memory vector store "
            "(data will not persist across restarts)"
        )
        return InMemoryVectorStore()


def create_vector_store(cfg, embedder) -> VectorStore:
    """Config-driven store construction (preferred over :func:`build_vector_store`).

    ``auto`` tries Chroma first and degrades to the in-memory store; an
    explicit ``chroma``/``memory`` request is honoured or fails loudly.
    """
    requested = (cfg.vector_backend or "auto").strip().lower()
    if requested == "memory":
        return InMemoryVectorStore()
    try:
        return ChromaVectorStore(cfg.vector_dir, cfg.vector_collection, embedder.identity)
    except VectorStoreError:
        if requested == "chroma":
            raise
        log.warning(
            "chromadb unavailable - using the in-memory vector store "
            "(data will not persist across restarts)"
        )
        return InMemoryVectorStore()
