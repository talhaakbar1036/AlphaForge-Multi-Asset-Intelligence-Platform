"""Composition root for the AI subsystem.

One place wires config -> embedder -> vector store -> registry -> ingestor ->
retriever -> LLM provider -> copilot, so the API routes and the Streamlit
dashboard share a single long-lived (lazy) instance instead of rebuilding the
embedding model on every request or rerun.

Nothing here is constructed at import time: the first call to
:func:`get_runtime` loads the (potentially heavy) embedding model. Call
:func:`reset_runtime` in tests after changing the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.ai.config import AIConfig, get_ai_config
from src.ai.copilot import FinancialCopilot
from src.ai.embeddings import EmbeddingBackend, get_embedder, reset_embedder_cache
from src.ai.ingest import DocumentIngestor
from src.ai.llm import LLMProvider, build_provider, provider_report
from src.ai.registry import DocumentRegistry
from src.ai.retrieval import Retriever
from src.ai.security import SlidingWindowRateLimiter
from src.ai.vectorstore import VectorStore, create_vector_store
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class AIRuntime:
    """All wired components for one configuration snapshot."""

    cfg: AIConfig
    embedder: EmbeddingBackend
    store: VectorStore
    registry: DocumentRegistry
    ingestor: DocumentIngestor
    retriever: Retriever
    provider: LLMProvider
    limiter: SlidingWindowRateLimiter
    copilot: FinancialCopilot

    def health(self) -> dict[str, Any]:
        """Secret-free component status for ``GET /ai/health``."""
        cfg = self.cfg
        if not cfg.enabled:
            return {
                "status": "disabled",
                "enabled": False,
                "config": cfg.as_public_dict(),
                "components": {
                    "vector_store": self.store.backend,
                    "embedder": self.embedder.identity,
                    "indexed_chunks": None,
                    "indexed_documents": None,
                    "llm": provider_report(cfg),
                },
            }
        try:
            chunks = self.store.count()
            documents = self.registry.count()
            index_error = None
        except Exception as exc:  # noqa: BLE001 - health must never raise
            chunks, documents, index_error = None, None, str(exc)
        return {
            "status": "ok" if index_error is None else "degraded",
            "enabled": True,
            "config": cfg.as_public_dict(),
            "components": {
                "vector_store": self.store.backend,
                "embedder": self.embedder.identity,
                "indexed_chunks": chunks,
                "indexed_documents": documents,
                "index_error": index_error,
                "llm": provider_report(cfg),
            },
        }


def build_runtime(cfg: AIConfig | None = None) -> AIRuntime:
    """Wire a fresh runtime (used directly by tests)."""
    cfg = cfg or get_ai_config()
    cfg.validate()
    cfg.vector_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(cfg)
    store = create_vector_store(cfg, embedder)
    registry = DocumentRegistry(cfg.vector_dir / "registry.sqlite3")
    ingestor = DocumentIngestor(cfg, store, registry, embedder)
    retriever = Retriever(cfg, store, embedder)
    provider = build_provider(cfg)
    limiter = SlidingWindowRateLimiter(cfg.rate_limit_per_minute)
    copilot = FinancialCopilot(cfg, retriever, provider)
    log.info(
        "AI runtime ready (store=%s, embedder=%s, llm=%s)",
        store.backend,
        embedder.identity,
        provider.name,
    )
    return AIRuntime(
        cfg=cfg,
        embedder=embedder,
        store=store,
        registry=registry,
        ingestor=ingestor,
        retriever=retriever,
        provider=provider,
        limiter=limiter,
        copilot=copilot,
    )


@lru_cache(maxsize=1)
def get_runtime() -> AIRuntime:
    """Process-wide runtime (embedding model loaded at most once)."""
    return build_runtime()


def reset_runtime() -> None:
    """Drop cached runtime/config/embedder (tests and post-env-change use)."""
    get_runtime.cache_clear()
    reset_embedder_cache()
    from src.ai.config import reset_ai_config

    reset_ai_config()
