"""Shared test fixtures.

The suite must never mutate the working tree: EDA figures and reports are
redirected to a per-test tmp directory, and the AI copilot runs entirely on
tmp dirs with the offline hashing embedder + in-memory vector store so no
model downloads, network calls or index pollution can happen.
"""
from __future__ import annotations

import pytest

from src.analysis import eda as eda_mod
from src.utils.config import CONFIG


@pytest.fixture(autouse=True)
def _isolate_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(eda_mod, "FIG_DIR", tmp_path / "figures" / "eda")
    monkeypatch.setattr(CONFIG, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(CONFIG, "figures_dir", tmp_path / "figures")


@pytest.fixture
def ai_env(tmp_path, monkeypatch):
    """Deterministic, offline, sandboxed AI configuration.

    Hashing embeddings (no model download), in-memory vector store (no
    chroma/index files in the repo), ``none`` provider (no network), and all
    state under ``tmp_path``. Resets the process-wide runtime caches around
    the test.
    """
    from src.ai.runtime import reset_runtime

    monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "true")
    monkeypatch.setenv("ALPHAFORGE_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("ALPHAFORGE_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("ALPHAFORGE_LLM_PROVIDER", "none")
    monkeypatch.setenv("ALPHAFORGE_VECTOR_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setenv("ALPHAFORGE_AI_DOCS_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("ALPHAFORGE_AI_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "")
    monkeypatch.setenv("ALPHAFORGE_LLM_RETRY_BACKOFF_S", "0.01")
    monkeypatch.delenv("ALPHAFORGE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def ai_runtime(ai_env):
    """A freshly built runtime on the sandboxed configuration."""
    from src.ai.runtime import build_runtime

    return build_runtime()
