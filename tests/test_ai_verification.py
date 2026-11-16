"""Live provider verification probes + the POST /ai/verify endpoint.

Provider HTTP is mocked throughout (the suite must run offline); the module
under test is exactly what a deployment runs to answer "does the real LLM
endpoint work here?".
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.ai.config import AIConfig
from src.ai.errors import LLMError, ProviderUnavailableError
from src.ai.llm import LLMProvider, NullProvider
from src.ai.runtime import reset_runtime
from src.ai.verification import (
    VerificationReport,
    verify_provider,
    verify_runtime,
)


def _cfg(**overrides) -> AIConfig:
    base = dict(llm_provider="none", llm_model="verify-model", llm_max_retries=0)
    base.update(overrides)
    return AIConfig(**base).validate()


class FakeProvider(LLMProvider):
    """Scripted provider: each answer comes from the next script entry."""

    name = "fake"
    supports_streaming = True

    def __init__(self, answers=None, available=True, raises=None, stream_deltas=()):
        self.answers = list(answers or ["OK"])
        self._available = available
        self._raises = raises
        self._stream_deltas = list(stream_deltas)
        self.calls: list[dict] = []

    def complete(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        self.calls.append({"prompt": prompt, "timeout_s": timeout_s})
        if self._raises is not None:
            raise self._raises
        return self.answers.pop(0) if self.answers else "OK"

    def stream(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        self.calls.append({"prompt": prompt, "timeout_s": timeout_s, "stream": True})
        yield from self._stream_deltas

    def is_available(self) -> bool:
        return self._available


class TestVerifyProvider:
    def test_all_green(self):
        provider = FakeProvider(answers=["OK", "4", "INSUFFICIENT EVIDENCE"], stream_deltas=["a", "b"])
        report = verify_provider(provider, _cfg())
        assert isinstance(report, VerificationReport)
        assert report.ok is True
        assert report.available is True
        assert report.provider == "fake"
        names = [p.name for p in report.probes]
        assert names == ["available", "responds", "instruction_following", "no_fabrication", "streaming"]
        assert all(p.ok for p in report.probes)
        assert report.probes[1].latency_ms is not None

    def test_unavailable_provider_fails_without_any_http(self):
        provider = FakeProvider(available=False)
        report = verify_provider(provider, _cfg())
        assert report.ok is False
        assert report.available is False
        assert "not reachable" in (report.error or "")
        assert [p.name for p in report.probes] == ["available"]
        assert provider.calls == []

    def test_null_provider_report(self):
        report = verify_provider(NullProvider(), _cfg())
        assert report.ok is False
        assert report.as_dict()["provider"] == "none"

    def test_advisory_probe_failure_is_not_fatal(self):
        # responds OK, but the model ignores the format instruction
        provider = FakeProvider(answers=["OK", "banana", "some text"], stream_deltas=["x"])
        report = verify_provider(provider, _cfg())
        advisory = {p.name: p for p in report.probes}
        assert advisory["responds"].ok is True
        assert advisory["instruction_following"].ok is False
        assert advisory["instruction_following"].required is False
        assert report.ok is True, "advisory probes must not fail the report"
        assert report.error is None

    def test_required_probe_failure_sets_error(self):
        provider = FakeProvider(answers=["", "4"], stream_deltas=["x"])  # empty = failure
        report = verify_provider(provider, _cfg())
        assert report.ok is False
        assert "responds" in (report.error or "")

    def test_probe_exception_is_captured_not_raised(self):
        provider = FakeProvider(raises=LLMError("boom"))
        report = verify_provider(provider, _cfg())
        assert report.ok is False
        assert any("boom" in p.detail for p in report.probes)

    def test_unreachable_probe_reports_typed_error(self):
        provider = FakeProvider(raises=ProviderUnavailableError("connection refused"))
        report = verify_provider(provider, _cfg())
        failed = [p for p in report.probes if not p.ok]
        assert failed
        assert any("connection refused" in p.detail for p in failed)

    def test_streaming_probe_skipped_on_request(self):
        provider = FakeProvider(answers=["OK", "4", "INSUFFICIENT EVIDENCE"], stream_deltas=["a"])
        report = verify_provider(provider, _cfg(), include_streaming=False)
        assert "streaming" not in [p.name for p in report.probes]
        assert report.ok is True

    def test_timeout_is_forwarded_to_probes(self):
        provider = FakeProvider(answers=["OK", "4", "INSUFFICIENT EVIDENCE"], stream_deltas=["a"])
        verify_provider(provider, _cfg(), timeout_s=7.5)
        assert provider.calls, "probes must call the provider"
        assert all(call["timeout_s"] == 7.5 for call in provider.calls)

    def test_non_streaming_provider_passes_streaming_probe_advisory(self):
        class NoStream(FakeProvider):
            supports_streaming = False

            def stream(self, *a, **k):  # pragma: no cover - must never be called
                raise AssertionError("stream() must not be called")

        provider = NoStream(answers=["OK", "4", "INSUFFICIENT EVIDENCE"])
        report = verify_provider(provider, _cfg())
        streaming = next(p for p in report.probes if p.name == "streaming")
        assert streaming.ok is True
        assert "does not support streaming" in streaming.detail

    def test_report_dict_is_json_safe_and_secret_free(self, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_LLM_API_KEY", "sk-canary-key")
        provider = FakeProvider(answers=["OK"], stream_deltas=["a"])
        report = verify_provider(provider, _cfg(llm_api_key="sk-canary-key"))
        blob = json.dumps(report.as_dict())
        assert "sk-canary-key" not in blob
        assert set(report.as_dict()) == {
            "provider", "model", "available", "ok", "checked_at", "probes", "error",
        }

    def test_verify_runtime_wrapper(self, ai_runtime):
        report = verify_runtime(ai_runtime)
        assert isinstance(report, VerificationReport)
        assert report.ok is False  # `none` provider in the test env
        assert report.provider == "none"


def _make_client() -> TestClient:
    from src.api.fastapi_app import create_app

    reset_runtime()
    return TestClient(create_app())


class TestVerifyEndpoint:
    def test_verify_returns_report_even_when_provider_down(self, ai_env):
        client = _make_client()
        resp = client.post("/ai/verify")
        # 200 + ok:false - the *endpoint* works, the provider answer is in the body
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["provider"] == "none"
        assert body["error"]

    def test_verify_uses_live_provider_when_available(self, ai_env, monkeypatch):
        from src.ai import verification

        client = _make_client()

        def _fake_verify(provider, cfg, **kwargs):
            # the route must pass through the *runtime's* provider
            assert provider.name == "none"  # fixture provider, reachable check mocked below
            return verification.VerificationReport(
                provider="fake",
                model="m",
                available=True,
                ok=True,
                probes=[verification.ProbeResult("responds", True, 1)],
            )

        monkeypatch.setattr(verification, "verify_provider", _fake_verify)
        resp = client.post("/ai/verify")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["probes"][0]["name"] == "responds"

    def test_verify_requires_admin_token_when_configured(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "topsecrettoken")
        client = _make_client()
        assert client.post("/ai/verify").status_code == 401
        ok = client.post("/ai/verify", headers={"Authorization": "Bearer topsecrettoken"})
        assert ok.status_code == 200

    def test_verify_disabled_returns_503(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "false")
        client = _make_client()
        resp = client.post("/ai/verify")
        assert resp.status_code == 503
        assert "ALPHAFORGE_AI_ENABLED" in resp.json()["detail"]

    def test_verify_does_not_leak_admin_token(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "canary-token-abc")
        client = _make_client()
        resp = client.post("/ai/verify", headers={"Authorization": "Bearer canary-token-abc"})
        assert resp.status_code == 200
        assert "canary-token-abc" not in resp.text
