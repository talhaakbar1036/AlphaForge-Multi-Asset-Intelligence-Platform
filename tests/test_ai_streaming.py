"""Streaming: provider streams, copilot.ask_stream events, POST /ai/chat/stream.

All provider HTTP is mocked (no network, no model server).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.ai.config import AIConfig
from src.ai.errors import (
    LLMError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from src.ai.llm import LLMProvider, NullProvider, OllamaProvider, OpenAIProvider
from src.ai.runtime import get_runtime, reset_runtime


def _cfg(**overrides) -> AIConfig:
    base = dict(
        llm_provider="none",
        llm_model="stream-model",
        ollama_base_url="http://127.0.0.1:9",
        llm_base_url="",
        llm_api_key="",
        llm_max_retries=0,
        llm_retry_backoff_s=0.0,
        llm_timeout_s=5.0,
    )
    base.update(overrides)
    return AIConfig(**base).validate()


class _StreamResp:
    def __init__(self, status=200, lines=(), text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._lines = list(lines)
        self.text = text or json.dumps({"error": "detail"})
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, str):
                yield line.encode("utf-8")
            else:
                yield line

    def json(self):
        try:
            return json.loads(self.text)
        except (TypeError, json.JSONDecodeError):
            return {}

    def close(self):
        self.closed = True


# ------------------------------------------------------------- providers --
class TestOllamaStream:
    def test_yields_ndjson_content(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, stream=False):
            captured.update(url=url, payload=json, stream=stream)
            return _StreamResp(
                lines=[
                    json_mod.dumps({"message": {"content": "Hello "}, "done": False}),
                    json_mod.dumps({"message": {"content": "world"}, "done": False}),
                    json_mod.dumps({"message": {"content": "!"}, "done": True}),
                ]
            )

        monkeypatch.setattr("src.ai.llm.requests.post", fake_post)
        provider = OllamaProvider("m", "http://127.0.0.1:11434", _cfg())
        deltas = list(provider.stream("hi"))
        assert deltas == ["Hello ", "world", "!"]
        assert captured["stream"] is True
        assert captured["payload"]["stream"] is True
        assert captured["url"].endswith("/api/chat")

    def test_error_chunk_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.post",
            lambda *a, **k: _StreamResp(lines=[json_mod.dumps({"error": "model not found"})]),
        )
        provider = OllamaProvider("m", "http://127.0.0.1:11434", _cfg())
        with pytest.raises(ProviderUnavailableError, match="model not found"):
            list(provider.stream("hi"))

    def test_empty_stream_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.post",
            lambda *a, **k: _StreamResp(lines=[json_mod.dumps({"done": True})]),
        )
        provider = OllamaProvider("m", "http://127.0.0.1:11434", _cfg())
        with pytest.raises(ProviderResponseError, match="without any content"):
            list(provider.stream("hi"))

    def test_http_error_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.post",
            lambda *a, **k: _StreamResp(status=500, text="server exploded"),
        )
        provider = OllamaProvider("m", "http://127.0.0.1:11434", _cfg())
        with pytest.raises(ProviderUnavailableError, match="500"):
            list(provider.stream("hi"))


class TestOpenAIStream:
    def test_yields_sse_deltas_until_done(self, monkeypatch):
        lines = [
            "data: " + json_mod.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
            "",
            "data: " + json_mod.dumps({"choices": [{"delta": {"content": "lo"}}]}),
            "data: [DONE]",
        ]
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _StreamResp(lines=lines))
        provider = OpenAIProvider("gpt-test", "https://api.example.com/v1", "sk-test", _cfg())
        assert list(provider.stream("hi")) == ["Hel", "lo"]

    def test_no_content_raises(self, monkeypatch):
        lines = ["data: " + json_mod.dumps({"choices": [{"delta": {}}]}), "data: [DONE]"]
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _StreamResp(lines=lines))
        provider = OpenAIProvider("gpt-test", "https://api.example.com/v1", "sk-test", _cfg())
        with pytest.raises(ProviderResponseError, match="without any content"):
            list(provider.stream("hi"))

    def test_missing_key_raises_before_network(self):
        provider = OpenAIProvider("gpt-test", "", "", _cfg())
        with pytest.raises(ProviderUnavailableError, match="no API key"):
            list(provider.stream("hi"))

    def test_auth_rejection(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.post",
            lambda *a, **k: _StreamResp(status=401, text="bad key"),
        )
        provider = OpenAIProvider("gpt-test", "https://api.example.com/v1", "sk-test", _cfg())
        with pytest.raises(ProviderUnavailableError, match="credentials"):
            list(provider.stream("hi"))


class TestProviderFallbacks:
    def test_null_provider_stream_raises(self):
        with pytest.raises(ProviderUnavailableError, match="streaming needs Ollama"):
            list(NullProvider().stream("hi"))

    def test_base_provider_streams_single_delta(self):
        class Minimal(LLMProvider):
            name = "minimal"

            def complete(self, prompt, **kwargs):
                return "one-shot answer"

            def is_available(self):
                return True

        assert list(Minimal().stream("hi")) == ["one-shot answer"]
        assert Minimal().supports_streaming is False

    def test_describe_reports_streaming_flag(self):
        assert NullProvider().describe()["supports_streaming"] is False
        assert OllamaProvider("m", "http://127.0.0.1:1", _cfg()).describe()[
            "supports_streaming"
        ] is True


# ------------------------------------------------------------- copilot -----
class _ScriptedProvider(LLMProvider):
    name = "scripted"
    supports_streaming = True

    def __init__(self, deltas=(), raise_after=0, raises=None, available=True):
        self.deltas = list(deltas)
        self.raise_after = raise_after
        self.raises = raises
        self._available = available
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return "".join(self.deltas) or "answer"

    def stream(self, prompt, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        for i, delta in enumerate(self.deltas):
            if i >= self.raise_after and self.raises is not None:
                raise self.raises
            yield delta
            if i + 1 >= self.raise_after and self.raises is not None:
                raise self.raises

    def is_available(self):
        return self._available


def _copilot(runtime, provider=None):
    from src.ai.copilot import FinancialCopilot

    if provider is None:
        return runtime.copilot
    return FinancialCopilot(runtime.cfg, runtime.retriever, provider=provider)


class TestAskStream:
    def test_event_order_and_final_matches_ask(self, ai_runtime):
        copilot = _copilot(ai_runtime)
        question = "What are the main risks for NVDA?"
        events = list(copilot.ask_stream(question))
        types = [e["type"] for e in events]
        assert types[0] == "meta"
        assert types[-1] == "final"
        assert "delta" in types
        assert "error" not in types

        meta = events[0]
        assert "sources" in meta
        assert "limitations" in meta
        assert meta["streaming"] is False  # `none` provider

        final = events[-1]
        text = "".join(e["text"] for e in events if e["type"] == "delta")
        assert final["answer"] == text
        assert copilot.ask(question).answer == final["answer"]

    def test_empty_question_raises(self, ai_runtime):
        with pytest.raises(ValueError, match="empty"):
            list(_copilot(ai_runtime).ask_stream("   "))

    def test_streaming_provider_deltas_reach_client(self, ai_runtime):
        provider = _ScriptedProvider(deltas=["Risk ", "one: export ", "controls."])
        copilot = _copilot(ai_runtime, provider)
        # Include a ticker so the engine supplies verified evidence; a question
        # with no document or metric evidence is correctly refused before the LLM.
        events = list(copilot.ask_stream("What are the risks for NVDA?"))
        deltas = [e["text"] for e in events if e["type"] == "delta"]
        assert deltas == ["Risk ", "one: export ", "controls."]
        final = events[-1]
        assert final["generated_by"] == "llm"
        assert final["answer"] == "Risk one: export controls."
        assert events[0]["streaming"] is True

    def test_midstream_failure_falls_back_without_fabrication(self, ai_runtime):
        provider = _ScriptedProvider(deltas=["partial "], raise_after=10, raises=LLMError("boom"))
        # raise immediately: no deltas delivered yet
        provider.raise_after = 0
        copilot = _copilot(ai_runtime, provider)
        events = list(copilot.ask_stream("What are the risks for NVDA?"))
        final = events[-1]
        assert final["type"] == "final"
        assert final["generated_by"] == "retrieval"
        assert any("LLM generation failed" in note for note in final["limitations"])
        assert final["answer"].strip(), "must still answer from retrieval"

    def test_error_event_is_terminal(self, ai_env):
        """Fatal generator errors must become a terminal SSE error frame."""
        client = _make_client()
        runtime = get_runtime()

        def boom(*a, **k):
            raise ValueError("question blew up")
            yield  # pragma: no cover

        runtime.copilot.ask_stream = boom
        response = client.post("/ai/chat/stream", json={"question": "x"})
        frames = _frames(response.text)
        payloads = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: {")]
        assert payloads[-1]["type"] == "error"
        assert frames[-1] == "data: [DONE]"


# ------------------------------------------------------------- endpoint ----
def _make_client() -> TestClient:
    from src.api.fastapi_app import create_app

    reset_runtime()
    return TestClient(create_app())


def _frames(body: str) -> list[str]:
    return [f for f in body.split("\n\n") if f.strip()]


class TestStreamEndpoint:
    def test_sse_shape(self, ai_env):
        client = _make_client()
        resp = client.post("/ai/chat/stream", json={"question": "What are the risks for NVDA?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"

        frames = _frames(resp.text)
        assert frames[-1] == "data: [DONE]"
        payloads = [json.loads(f[len("data: "):]) for f in frames[:-1]]
        assert payloads[0]["type"] == "meta"
        assert payloads[-1]["type"] == "final"
        assert any(p["type"] == "delta" for p in payloads)

    def test_final_frame_matches_blocking_chat(self, ai_env):
        client = _make_client()
        question = "What are the main risks for NVDA?"
        streamed = client.post("/ai/chat/stream", json={"question": question})
        blocking = client.post("/ai/chat", json={"question": question})
        assert streamed.status_code == blocking.status_code == 200

        final = [
            json.loads(f[len("data: "):])
            for f in _frames(streamed.text)
            if f != "data: [DONE]"
        ][-1]
        assert final["answer"] == blocking.json()["answer"]
        assert final["category"] == blocking.json()["category"]

    def test_text_is_reassemblable_from_deltas(self, ai_env):
        client = _make_client()
        resp = client.post("/ai/chat/stream", json={"question": "Summarise NVDA risks"})
        payloads = [
            json.loads(f[len("data: "):]) for f in _frames(resp.text) if f != "data: [DONE]"
        ]
        text = "".join(p["text"] for p in payloads if p["type"] == "delta")
        final = [p for p in payloads if p["type"] == "final"][0]
        assert text == final["answer"]

    def test_validation_error(self, ai_env):
        resp = _make_client().post("/ai/chat/stream", json={"question": ""})
        assert resp.status_code == 422

    def test_disabled_returns_503(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "false")
        resp = _make_client().post("/ai/chat/stream", json={"question": "hello"})
        assert resp.status_code == 503
        assert "ALPHAFORGE_AI_ENABLED" in resp.json()["detail"]

    def test_midstream_error_still_terminates_cleanly(self, ai_env, monkeypatch):
        client = _make_client()
        runtime = get_runtime()

        def boom(*a, **k):
            raise RuntimeError("stream exploded")
            yield  # pragma: no cover

        monkeypatch.setattr(runtime.copilot, "ask_stream", boom)
        resp = client.post("/ai/chat/stream", json={"question": "hello"})
        assert resp.status_code == 200
        frames = _frames(resp.text)
        assert frames[-1] == "data: [DONE]"
        payloads = [json.loads(f[len("data: "):]) for f in frames[:-1]]
        assert payloads[0]["type"] == "error"
        assert "internal error" in payloads[0]["message"]

    def test_rate_limited_request_is_rejected(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_RATE_LIMIT_PER_MIN", "1")
        reset_runtime()
        client = _make_client()
        assert client.post("/ai/chat/stream", json={"question": "one"}).status_code == 200
        second = client.post("/ai/chat/stream", json={"question": "two"})
        assert second.status_code == 429


json_mod = json
