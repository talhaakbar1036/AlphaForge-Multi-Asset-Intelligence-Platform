"""LLM provider layer: availability, retries, timeouts, malformed output.

All HTTP is mocked - the suite must pass with no network and no provider.
"""
from __future__ import annotations

import json

import pytest
import requests

from src.ai.config import AIConfig
from src.ai.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from src.ai.llm import NullProvider, OllamaProvider, OpenAIProvider, build_provider, provider_report


def _cfg(**overrides) -> AIConfig:
    base = dict(
        llm_provider="none",
        llm_model="test-model",
        ollama_base_url="http://127.0.0.1:9",  # unroutable port
        llm_base_url="",
        llm_api_key="",
        llm_max_retries=0,
        llm_retry_backoff_s=0.0,
        llm_timeout_s=5.0,
    )
    base.update(overrides)
    return AIConfig(**base).validate()


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestNullProvider:
    def test_unavailable_and_raises(self):
        p = NullProvider()
        assert p.is_available() is False
        with pytest.raises(ProviderUnavailableError):
            p.complete("hello")
        assert p.describe()["provider"] == "none"


class TestBuildProvider:
    def test_none(self):
        assert isinstance(build_provider(_cfg(llm_provider="none")), NullProvider)

    def test_ollama_selected(self):
        p = build_provider(_cfg(llm_provider="ollama"))
        assert isinstance(p, OllamaProvider)
        assert p.model == "test-model"

    def test_openai_selected_with_key(self):
        p = build_provider(_cfg(llm_provider="openai", llm_api_key="sk-test"))
        assert isinstance(p, OpenAIProvider)
        assert p.is_available() is True

    def test_openai_without_key_is_unavailable(self):
        p = build_provider(_cfg(llm_provider="openai"))
        assert p.is_available() is False
        with pytest.raises(ProviderUnavailableError):
            p.complete("hi")

    def test_auto_falls_back_to_none_when_nothing_reachable(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.get",
            lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
        )
        p = build_provider(_cfg(llm_provider="auto"))
        assert isinstance(p, NullProvider)

    def test_auto_prefers_ollama_when_healthy(self, monkeypatch):
        monkeypatch.setattr("src.ai.llm.requests.get", lambda *a, **k: _Resp(200, {"models": []}))
        p = build_provider(_cfg(llm_provider="auto"))
        assert isinstance(p, OllamaProvider)


class TestOllamaProvider:
    def test_availability(self, monkeypatch):
        p = OllamaProvider("m", "http://localhost:11434", _cfg())
        monkeypatch.setattr("src.ai.llm.requests.get", lambda *a, **k: _Resp(200, {}))
        assert p.is_available() is True
        monkeypatch.setattr(
            "src.ai.llm.requests.get",
            lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
        )
        assert p.is_available() is False

    def test_complete_success(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["timeout"] = timeout
            return _Resp(200, {"message": {"content": "  grounded answer  "}})

        monkeypatch.setattr("src.ai.llm.requests.post", fake_post)
        p = OllamaProvider("m", "http://localhost:11434", _cfg())
        out = p.complete("prompt", system="sys", max_tokens=64, temperature=0.2)
        assert out == "grounded answer"
        assert captured["url"].endswith("/api/chat")
        assert captured["payload"]["stream"] is False
        assert captured["payload"]["options"]["num_predict"] == 64
        assert captured["payload"]["messages"][0] == {"role": "system", "content": "sys"}

    def test_timeout_maps_to_typed_error(self, monkeypatch):
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise requests.Timeout("slow")

        monkeypatch.setattr("src.ai.llm.requests.post", boom)
        p = OllamaProvider("m", "http://x", _cfg(llm_max_retries=1))
        with pytest.raises(ProviderTimeoutError):
            p.complete("q")
        assert calls["n"] == 2  # initial + one retry

    def test_malformed_payload(self, monkeypatch):
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _Resp(200, {"unexpected": 1}))
        p = OllamaProvider("m", "http://x", _cfg())
        with pytest.raises(ProviderResponseError):
            p.complete("q")

    def test_non_json_payload(self, monkeypatch):
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _Resp(200, None, text="<html>"))
        p = OllamaProvider("m", "http://x", _cfg())
        with pytest.raises(ProviderResponseError):
            p.complete("q")

    def test_server_error_exhausts_retries(self, monkeypatch):
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _Resp(500, {"error": "boom"}))
        p = OllamaProvider("m", "http://x", _cfg(llm_max_retries=1))
        with pytest.raises(ProviderUnavailableError) as excinfo:
            p.complete("q")
        assert "500" in str(excinfo.value)


class TestOpenAIProvider:
    def test_auth_rejection_no_retry(self, monkeypatch):
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return _Resp(401, {"error": {"message": "bad key"}})

        monkeypatch.setattr("src.ai.llm.requests.post", fake)
        p = OpenAIProvider("m", "http://api", "sk-wrong", _cfg(llm_max_retries=3))
        with pytest.raises(ProviderUnavailableError):
            p.complete("q")
        assert calls["n"] == 1  # auth failures are not retried

    def test_success_extracts_content(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.post",
            lambda *a, **k: _Resp(200, {"choices": [{"message": {"content": "answer"}}]}),
        )
        p = OpenAIProvider("m", "http://api", "sk-ok", _cfg())
        assert p.complete("q") == "answer"

    def test_malformed_choices(self, monkeypatch):
        monkeypatch.setattr("src.ai.llm.requests.post", lambda *a, **k: _Resp(200, {"choices": []}))
        p = OpenAIProvider("m", "http://api", "sk-ok", _cfg())
        with pytest.raises(ProviderResponseError):
            p.complete("q")

    def test_missing_key_raises_before_network(self):
        p = OpenAIProvider("m", "http://api", "", _cfg())
        with pytest.raises(ProviderUnavailableError):
            p.complete("q")

    def test_describe_never_leaks_key(self):
        p = OpenAIProvider("m", "http://api", "sk-secret-123", _cfg())
        payload = json.dumps(p.describe())
        assert "sk-secret-123" not in payload
        assert p.describe()["api_key_configured"] is True


class TestProviderReport:
    def test_report_has_no_secrets(self, monkeypatch):
        monkeypatch.setattr(
            "src.ai.llm.requests.get",
            lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()),
        )
        cfg = _cfg(llm_provider="auto", llm_api_key="sk-super-secret", llm_base_url="http://host")
        report = provider_report(cfg)
        assert "sk-super-secret" not in json.dumps(report)
        assert report["configured"] == "auto"
        assert report["hosted"]["api_key_configured"] is True
        assert report["ollama"]["reachable"] is False
