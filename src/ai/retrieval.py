"""Retrieval: semantic search + metadata filters + optional keyword rerank.

The pipeline is deliberately simple:

    embed(query) -> vector search (+ metadata filter) -> hybrid rerank
        -> deduplicate -> top-k chunks with citations

No agent framework, no query planner - deterministic routing decides *what*
to retrieve (see :mod:`src.ai.routing`), this module only does the search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.ai.config import AIConfig
from src.ai.errors import RetrievalError
from src.ai.vectorstore import (
    KEY_DOC_ID,
    KEY_DOC_TYPE,
    KEY_PAGE,
    KEY_PUB_TS,
    KEY_TICKER,
    KEY_VISIBILITY,
    SearchHit,
    VectorStore,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_CANDIDATES = 60


@dataclass
class RetrievalFilters:
    """Metadata constraints applied *before* ranking."""

    tickers: list[str] | None = None
    doc_types: list[str] | None = None
    published_from: str | None = None
    published_to: str | None = None
    document_ids: list[str] | None = None
    include_private: bool = False

    def to_where(self) -> dict[str, Any]:
        where: dict[str, Any] = {}
        if self.include_private:
            where[KEY_VISIBILITY] = ["public", "private"]
        else:
            where[KEY_VISIBILITY] = "public"
        if self.tickers:
            where[KEY_TICKER] = [t.upper() for t in self.tickers]
        if self.doc_types:
            where[KEY_DOC_TYPE] = list(self.doc_types)
        if self.document_ids:
            where[KEY_DOC_ID] = list(self.document_ids)
        range_clause: dict[str, Any] = {}
        if self.published_from:
            range_clause["gte"] = _to_ts(self.published_from, end_of_day=False)
        if self.published_to:
            range_clause["lte"] = _to_ts(self.published_to, end_of_day=True)
        if range_clause:
            where[f"{KEY_PUB_TS}_range"] = range_clause
        return where

    def describe(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers,
            "doc_types": self.doc_types,
            "published_from": self.published_from,
            "published_to": self.published_to,
            "document_ids": self.document_ids,
            "include_private": self.include_private,
        }


def _to_ts(value: str, end_of_day: bool) -> float:
    raw = str(value).strip()
    try:
        if len(raw) == 4:  # year only
            dt = datetime(int(raw), 1, 1, tzinfo=timezone.utc)
            if end_of_day:
                dt = datetime(int(raw), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        else:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
    except ValueError as exc:
        raise RetrievalError(f"invalid date filter {value!r} (expected YYYY-MM-DD)") from exc
    return dt.timestamp()


@dataclass
class RetrievedChunk:
    """A chunk that survived ranking, with everything needed for a citation."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    title: str | None = None
    filename: str | None = None
    ticker: str | None = None
    doc_type: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        page = f", p.{self.page_number}" if self.page_number else ""
        date = f" ({self.publication_date})" if self.publication_date else ""
        return f"[{self.rank}] {self.title or self.filename or self.document_id}{date}{page}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "filename": self.filename,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "page_number": self.page_number,
            "score": round(float(self.score), 4),
            "citation": self.citation,
            "text_preview": self.text[:400],
        }


def _terms(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _keyword_score(query: str, text: str) -> float:
    """Cheap lexical overlap score in [0, 1] (no corpus statistics needed)."""
    q = _terms(query)
    if not q:
        return 0.0
    t = _terms(text)
    if not t:
        return 0.0
    overlap = q & t
    if not overlap:
        return 0.0
    return len(overlap) / len(q) * 0.6 + len(overlap) / max(len(t), 1) * 0.4


def _deduplicate(hits: list[SearchHit]) -> list[SearchHit]:
    """Drop repeated text (overlapping chunks) and duplicate pages."""
    seen: set[str] = set()
    out: list[SearchHit] = []
    for hit in hits:
        fingerprint = " ".join(hit.text.lower().split())[:600]
        key = hit.id if len(fingerprint) > 120 else fingerprint or hit.id
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


class Retriever:
    """Config-bound retrieval facade used by the copilot and ``/ai/retrieve``."""

    def __init__(self, cfg: AIConfig, store: VectorStore, embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        hybrid: bool | None = None,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []
        k = int(top_k or self.cfg.retrieval_top_k)
        k = max(1, min(k, 50))
        filters = filters or RetrievalFilters()
        hybrid = self.cfg.retrieval_hybrid if hybrid is None else bool(hybrid)

        try:
            vector = self.embedder.encode([query])[0]
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"could not embed the query: {exc}") from exc

        where = filters.to_where()
        try:
            hits = self.store.query(vector, k=min(k * 3, MAX_CANDIDATES) if hybrid else k, where=where)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"vector search failed: {exc}") from exc
        if not hits:
            return []

        if hybrid:
            blended = []
            max_rank = max(len(hits), 1)
            for position, hit in enumerate(hits):
                semantic = hit.score
                lexical = _keyword_score(query, hit.text)
                # rank-aware blend: semantic dominates, keywords break ties
                rank_prior = 1.0 - (position / max_rank) * 0.3
                blended.append((0.75 * semantic + 0.25 * lexical * rank_prior, hit))
            blended.sort(key=lambda pair: pair[0], reverse=True)
            hits = [hit for _, hit in blended[:k]]
        else:
            hits = hits[:k]

        hits = _deduplicate(hits)
        return [
            RetrievedChunk(
                chunk_id=str(hit.metadata.get("chunk_id") or hit.id),
                document_id=str(hit.metadata.get(KEY_DOC_ID) or ""),
                text=hit.text,
                score=hit.score,
                rank=i + 1,
                title=hit.metadata.get("title"),
                filename=hit.metadata.get("filename"),
                ticker=hit.metadata.get(KEY_TICKER),
                doc_type=hit.metadata.get("doc_type"),
                publication_date=hit.metadata.get("publication_date"),
                source_url=hit.metadata.get("source_url"),
                page_number=hit.metadata.get(KEY_PAGE),
                metadata=dict(hit.metadata),
            )
            for i, hit in enumerate(hits)
        ]
