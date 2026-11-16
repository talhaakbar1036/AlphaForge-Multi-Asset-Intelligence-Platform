"""FastAPI /ai endpoints: status codes, validation, auth, health hygiene."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src.ai.runtime import get_runtime, reset_runtime

DOC_TEXT = (
    "NVIDIA annual report risk factors. Export controls restrict advanced "
    "GPU sales to certain markets. Demand concentration among hyperscale "
    "customers is a material risk."
)


def make_client() -> TestClient:
    """Fresh app + runtime bound to the current (monkeypatched) environment."""
    from src.api.fastapi_app import create_app

    get_runtime()  # prime the cached runtime with the test configuration
    return TestClient(create_app())


@pytest.fixture
def client(ai_env):
    return make_client()


def _upload(client: TestClient, text: str = DOC_TEXT, name: str = "NVDA_10-K.txt",
            token: str | None = None, **form):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/ai/documents/upload",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/plain")},
        data=form,
        headers=headers,
    )


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["enabled"] is True
        assert body["components"]["vector_store"] == "memory"
        assert body["components"]["embedder"].startswith("hashing:")
        assert body["components"]["indexed_chunks"] == 0
        assert body["components"]["llm"]["configured"] == "none"

    def test_health_never_leaks_secrets(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "topsecrettoken")
        monkeypatch.setenv("ALPHAFORGE_LLM_API_KEY", "sk-canary-123")
        reset_runtime()
        resp = make_client().get("/ai/health")
        assert resp.status_code == 200
        text = resp.text
        assert "topsecrettoken" not in text
        assert "sk-canary-123" not in text
        assert resp.json()["config"]["auth_required"] is True

    def test_disabled_copilot_health_still_works(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "false")
        reset_runtime()
        client = make_client()
        assert client.get("/ai/health").json()["status"] == "disabled"
        assert client.post("/ai/chat", json={"question": "hi"}).status_code == 503
        assert client.get("/ai/documents").status_code == 503
        # quantitative API is untouched
        assert client.get("/api/v1/health").status_code == 200


class TestChat:
    def test_chat_success(self, client):
        resp = client.post("/ai/chat", json={"question": "What are NVDA's disclosed risks?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["generated_by"] == "retrieval"
        assert "sources" in body and "limitations" in body

    def test_chat_with_market_metrics(self, client):
        resp = client.post(
            "/ai/chat",
            json={"question": "Explain NVDA's measured risk", "ticker": "NVDA",
                  "include_market_metrics": True},
        )
        assert resp.status_code == 200
        metrics = resp.json()["market_metrics"]
        assert metrics["symbol"] == "NVDA"
        assert metrics["decision"] is not None or metrics["limitations"]

    def test_chat_validation_errors(self, client):
        assert client.post("/ai/chat", json={"question": ""}).status_code == 422
        assert client.post("/ai/chat", json={}).status_code == 422
        assert client.post(
            "/ai/chat", json={"question": "x" * 4001}
        ).status_code == 422

    def test_method_not_allowed(self, client):
        assert client.get("/ai/chat").status_code == 405
        assert client.delete("/ai/chat").status_code == 405

    def test_rate_limit(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_RATE_LIMIT_PER_MIN", "2")
        reset_runtime()
        client = make_client()
        payload = {"question": "hello", "include_market_metrics": False}
        assert client.post("/ai/chat", json=payload).status_code == 200
        assert client.post("/ai/chat", json=payload).status_code == 200
        assert client.post("/ai/chat", json=payload).status_code == 429


class TestDocuments:
    def test_upload_index_list_delete_cycle(self, client):
        resp = _upload(client, ticker="NVDA", title="NVIDIA 10-K")
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "indexed"
        assert body["chunks"] >= 1
        doc_id = body["document_id"]

        listing = client.get("/ai/documents").json()
        assert listing["count"] == 1
        assert listing["documents"][0]["title"] == "NVIDIA 10-K"
        assert "stored_path" not in listing["documents"][0]

        deleted = client.delete(f"/ai/documents/{doc_id}")
        assert deleted.status_code == 200
        assert client.get("/ai/documents").json()["count"] == 0
        assert client.delete(f"/ai/documents/{doc_id}").status_code == 404

    def test_upload_rejects_unsupported_type(self, client):
        resp = client.post(
            "/ai/documents/upload",
            files={"file": ("app.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_sanitises_traversal_filename(self, client):
        # path components are stripped, never written outside the doc dir
        resp = _upload(client, name="../../NVDA_10-K.txt")
        assert resp.status_code == 201
        assert resp.json()["filename"] == "NVDA_10-K.txt"

    def test_upload_rejects_empty_filename(self, client):
        # an empty filename never reaches the ingestor: the multipart part is
        # not treated as a file, so request validation rejects it
        resp = client.post(
            "/ai/documents/upload",
            files={"file": ("", io.BytesIO(b"content"), "text/plain")},
        )
        assert resp.status_code == 422

    def test_upload_rejects_oversize(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_MAX_UPLOAD_MB", "1")
        reset_runtime()
        client = make_client()
        payload = "x" * (1024 * 1024 + 10)
        resp = _upload(client, text=payload, name="big.txt")
        assert resp.status_code == 413

    def test_upload_requires_token_when_configured(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "sekrit")
        reset_runtime()
        client = make_client()
        assert _upload(client).status_code == 401
        assert _upload(client, token="wrong").status_code == 401
        assert _upload(client, token="sekrit").status_code == 201

    def test_delete_requires_token_when_configured(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "sekrit")
        reset_runtime()
        client = make_client()
        doc_id = _upload(client, token="sekrit").json()["document_id"]
        assert client.delete(f"/ai/documents/{doc_id}").status_code == 401
        assert client.delete(
            f"/ai/documents/{doc_id}", headers={"Authorization": "Bearer sekrit"}
        ).status_code == 200

    def test_invalid_document_id_rejected(self, client):
        assert client.delete("/ai/documents/not%20a%20valid%20id").status_code == 400


class TestRetrieve:
    def test_retrieve_returns_chunks(self, client):
        _upload(client, ticker="NVDA")
        resp = client.post("/ai/retrieve", json={"query": "export controls GPU", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert body["chunks"][0]["ticker"] == "NVDA"
        assert body["chunks"][0]["text_preview"]

    def test_retrieve_validation(self, client):
        assert client.post("/ai/retrieve", json={"query": ""}).status_code == 422
        assert client.post(
            "/ai/retrieve", json={"query": "risk", "published_from": "bad-date"}
        ).status_code == 422
        assert client.post(
            "/ai/retrieve", json={"query": "risk", "top_k": 999}
        ).status_code == 422

    def test_retrieve_empty_index(self, client):
        resp = client.post("/ai/retrieve", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestScenarios:
    def test_catalogue(self, client):
        resp = client.get("/ai/scenarios")
        assert resp.status_code == 200
        assert "price_shock" in resp.json()["scenarios"]
        assert "never a forecast" in resp.json()["note"]

    def test_price_shock_via_api(self, client):
        resp = client.post(
            "/ai/scenarios",
            json={"scenario": "price_shock", "params": {"symbol": "NVDA", "shock_pct": -10}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "hypothetical"
        assert body["results"]["portfolio_value_change_pct"] == pytest.approx(-10.0, abs=0.01)

    def test_unknown_scenario_is_400(self, client):
        resp = client.post("/ai/scenarios", json={"scenario": "nope"})
        assert resp.status_code == 400
        assert "unknown scenario" in resp.json()["detail"]

    def test_invalid_params_is_400(self, client):
        resp = client.post(
            "/ai/scenarios",
            json={"scenario": "price_shock", "params": {"symbol": "NOPE", "shock_pct": -5}},
        )
        assert resp.status_code == 400
