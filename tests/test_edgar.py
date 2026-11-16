"""SEC EDGAR auto-fetch: fair-access client, caching, ingestion, API route.

All HTTP is mocked - the suite must pass with no network. The mocks mirror
the real SEC payloads (company_tickers.json + submissions.json) so parsing is
exercised against realistic shapes.
"""
from __future__ import annotations

import json
import time

import pytest
import requests
from fastapi.testclient import TestClient

from src.ai.config import AIConfig
from src.ai.edgar import (
    EdgarClient,
    EdgarFetchResult,
    _MinIntervalThrottle,
    _parse_ticker_map,
    ingest_latest_filings,
)
from src.ai.errors import EdgarError
from src.ai.runtime import reset_runtime

TICKER_MAP = {
    "fields": ["cik", "ticker", "title"],
    "data": [[1045810, "NVDA", "NVIDIA CORP"], [320193, "AAPL", "APPLE INC"]],
}
LEGACY_TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"}}

SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "10-K", "10-Q"],
            "accessionNumber": ["0001-24-0001", "0001-24-0002", "0001-23-0009", "0001-24-0003"],
            "filingDate": ["2024-08-01", "2024-07-15", "2023-11-01", "2024-05-02"],
            "reportDate": ["2024-07-28", "", "2023-10-29", "2024-04-28"],
            "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
        }
    }
}


class _Resp:
    def __init__(self, status=200, payload=None, content: bytes | None = None):
        self.status_code = status
        self.ok = 200 <= status < 300
        if content is not None:
            self.content = content
        elif payload is not None:
            self.content = json.dumps(payload).encode("utf-8")
        else:
            self.content = b""

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")


def _cfg(tmp_path, **overrides) -> AIConfig:
    base = dict(
        edgar_enabled=True,
        edgar_user_agent="AlphaForge-AI/0.2 (tests; contact: tests@example.com)",
        edgar_min_interval_s=0.05,
        edgar_cache_dir=tmp_path / "edgar_cache",
        edgar_max_filings=5,
        llm_provider="none",
    )
    base.update(overrides)
    return AIConfig(**base).validate()


def _install_http(monkeypatch, responses: dict[str, object]):
    """Route ``requests.get`` by URL substring; count calls per URL."""
    calls: list[str] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                if callable(value):
                    return value(url, headers=headers, timeout=timeout)
                return value
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr("src.ai.edgar.requests.get", fake_get)
    return calls


class TestTickerMap:
    def test_parse_new_shape(self):
        assert _parse_ticker_map(TICKER_MAP) == {"NVDA": 1045810, "AAPL": 320193}

    def test_parse_legacy_shape(self):
        assert _parse_ticker_map(LEGACY_TICKER_MAP) == {"AAPL": 320193}

    def test_parse_garbage(self):
        with pytest.raises(EdgarError, match="shape"):
            _parse_ticker_map(["not", "a", "map"])

    def test_cik_resolution_and_disk_cache(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        calls = _install_http(monkeypatch, {"company_tickers.json": _Resp(payload=TICKER_MAP)})

        client = EdgarClient(cfg)
        assert client.cik_for("nvda") == 1045810
        assert len(calls) == 1

        # second lookup hits the in-memory map
        assert client.cik_for("AAPL") == 320193
        assert len(calls) == 1

        # a *new* client reads the on-disk cache - zero network calls
        calls.clear()
        fresh = EdgarClient(cfg)
        assert fresh.cik_for("NVDA") == 1045810
        assert calls == []
        assert (cfg.edgar_cache_dir / "company_tickers.json").is_file()

    def test_unknown_ticker_is_helpful(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(monkeypatch, {"company_tickers.json": _Resp(payload=TICKER_MAP)})
        with pytest.raises(EdgarError, match="company_tickers.json"):
            EdgarClient(cfg).cik_for("ZZZZ")

    def test_empty_ticker_rejected(self, tmp_path):
        with pytest.raises(EdgarError, match="ticker is required"):
            EdgarClient(_cfg(tmp_path)).cik_for("  ")


class TestListFilings:
    def _client(self, tmp_path, monkeypatch) -> tuple[EdgarClient, list]:
        cfg = _cfg(tmp_path)
        calls = _install_http(
            monkeypatch,
            {
                "company_tickers.json": _Resp(payload=TICKER_MAP),
                "data.sec.gov/submissions": _Resp(payload=SUBMISSIONS),
            },
        )
        return EdgarClient(cfg), calls

    def test_filters_to_configured_forms(self, tmp_path, monkeypatch):
        client, _ = self._client(tmp_path, monkeypatch)
        filings = client.list_filings("NVDA")
        assert [f.form for f in filings] == ["10-Q", "8-K", "10-K", "10-Q"]  # default forms incl. 8-K
        assert filings[0].primary_document_url.startswith(
            "https://www.sec.gov/Archives/edgar/data/1045810/"
        )

    def test_count_limits_results(self, tmp_path, monkeypatch):
        client, _ = self._client(tmp_path, monkeypatch)
        assert len(client.list_filings("NVDA", forms=["10-Q"], count=1)) == 1

    def test_explicit_forms_override_default(self, tmp_path, monkeypatch):
        client, _ = self._client(tmp_path, monkeypatch)
        filings = client.list_filings("AAPL", forms=["8-K"])
        assert [f.form for f in filings] == ["8-K"]

    def test_submissions_payload_is_required(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(
            monkeypatch,
            {
                "company_tickers.json": _Resp(payload=TICKER_MAP),
                "data.sec.gov/submissions": _Resp(content=b"not json"),
            },
        )
        with pytest.raises(EdgarError, match="not valid JSON"):
            EdgarClient(cfg).list_filings("NVDA")


class TestHttpFailures:
    def test_forbidden_mentions_user_agent(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path, edgar_user_agent="AlphaForge-AI/0.2")
        _install_http(monkeypatch, {"company_tickers.json": _Resp(status=403)})
        with pytest.raises(EdgarError, match="ALPHAFORGE_EDGAR_USER_AGENT"):
            EdgarClient(cfg).cik_for("NVDA")

    def test_not_found(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(monkeypatch, {"company_tickers.json": _Resp(status=404)})
        with pytest.raises(EdgarError, match="404"):
            EdgarClient(cfg).cik_for("NVDA")

    def test_timeout_is_typed(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(monkeypatch, {"company_tickers.json": requests.Timeout("slow")})
        with pytest.raises(EdgarError, match="timed out"):
            EdgarClient(cfg).cik_for("NVDA")

    def test_unreachable_is_typed(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(monkeypatch, {"company_tickers.json": requests.ConnectionError("down")})
        with pytest.raises(EdgarError, match="unreachable"):
            EdgarClient(cfg).cik_for("NVDA")

    def test_disabled_short_circuits_before_any_request(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path, edgar_enabled=False)
        calls = _install_http(monkeypatch, {"company_tickers.json": _Resp(payload=TICKER_MAP)})
        with pytest.raises(EdgarError, match="ALPHAFORGE_EDGAR_ENABLED"):
            EdgarClient(cfg).cik_for("NVDA")
        assert calls == []

    def test_user_agent_header_is_descriptive(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        seen = {}

        def capture(url, headers=None, timeout=None):
            seen.update(headers or {})
            return _Resp(payload=TICKER_MAP)

        _install_http(monkeypatch, {"company_tickers.json": capture})
        EdgarClient(cfg).cik_for("NVDA")
        assert "contact:" in seen["User-Agent"]


class TestThrottle:
    def test_first_wait_is_free_later_one_paces(self):
        throttle = _MinIntervalThrottle(0.05)
        throttle.wait()
        started = time.monotonic()
        throttle.wait()
        assert time.monotonic() - started >= 0.04, "min interval must be enforced"

    def test_zero_interval_never_sleeps(self):
        throttle = _MinIntervalThrottle(0.0)
        started = time.monotonic()
        throttle.wait()
        throttle.wait()
        assert time.monotonic() - started < 0.04


class TestIngestion:
    def test_ingest_latest_filings_captures_provenance(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(
            monkeypatch,
            {
                "company_tickers.json": _Resp(payload=TICKER_MAP),
                "data.sec.gov/submissions": _Resp(payload=SUBMISSIONS),
                "Archives/edgar": _Resp(content=b"<html><body>NVIDIA 10-Q filing</body></html>"),
            },
        )

        seen: list[dict] = []

        class FakeIngestor:
            def ingest_bytes(self, data, filename, **meta):
                seen.append({"data": data, "filename": filename, **meta})
                from src.ai.ingest import IngestResult

                return IngestResult(
                    document_id="doc-1",
                    filename=filename,
                    status="indexed",
                    chunks=3,
                )

        result = ingest_latest_filings(FakeIngestor(), cfg, "NVDA", count=1)
        assert isinstance(result, EdgarFetchResult)
        assert result.ingested_count == 1
        assert result.filings_found == 1
        meta = seen[0]
        assert meta["ticker"] == "NVDA"
        assert meta["doc_type"] == "10-Q"
        assert meta["publication_date"] == "2024-08-01"
        assert meta["source_url"].startswith("https://www.sec.gov/Archives/")
        assert "NVDA_10-Q_2024-08-01" in meta["filename"]
        assert result.details[0]["ingested"] is True

    def test_one_bad_filing_does_not_kill_the_batch(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _install_http(
            monkeypatch,
            {
                "company_tickers.json": _Resp(payload=TICKER_MAP),
                "data.sec.gov/submissions": _Resp(payload=SUBMISSIONS),
                "Archives/edgar": _Resp(status=500),
            },
        )

        class FakeIngestor:
            def ingest_bytes(self, *a, **k):  # pragma: no cover - never reached
                raise AssertionError("should not ingest")

        result = ingest_latest_filings(FakeIngestor(), cfg, "NVDA", count=2)
        assert result.ingested_count == 0
        assert result.skipped_count == 2
        assert all("HTTP 500" in line for line in result.skipped)

    def test_no_matching_filings_reports_skip(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        empty = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001"],
                    "filingDate": ["2024-01-01"],
                    "reportDate": [""],
                    "primaryDocument": ["x.htm"],
                }
            }
        }
        _install_http(
            monkeypatch,
            {
                "company_tickers.json": _Resp(payload=TICKER_MAP),
                "data.sec.gov/submissions": _Resp(payload=empty),
            },
        )
        result = ingest_latest_filings(object(), cfg, "NVDA")
        assert result.ingested_count == 0
        assert result.skipped == ["no filings matched the requested forms"]


class TestEdgarConfig:
    def test_env_parsing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ALPHAFORGE_EDGAR_ENABLED", "true")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_FORMS", "10-K,8-K")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_MAX_FILINGS", "3")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_MIN_INTERVAL_S", "0.5")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_CACHE_DIR", str(tmp_path / "cache"))
        cfg = AIConfig.from_env()
        assert cfg.edgar_enabled is True
        assert cfg.edgar_forms == "10-K,8-K"
        assert cfg.edgar_max_filings == 3
        assert cfg.edgar_min_interval_s == 0.5

    def test_validation_bounds(self, tmp_path):
        with pytest.raises(Exception, match="ALPHAFORGE_EDGAR_MIN_INTERVAL_S"):
            _cfg(tmp_path, edgar_min_interval_s=0.001)
        with pytest.raises(Exception, match="ALPHAFORGE_EDGAR_MAX_FILINGS"):
            _cfg(tmp_path, edgar_max_filings=99)

    def test_public_dict_exposes_edgar_settings(self, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_EDGAR_ENABLED", "false")
        public = AIConfig.from_env().as_public_dict()
        assert public["edgar_enabled"] is False
        assert public["edgar_forms"]


def _make_client() -> TestClient:
    from src.api.fastapi_app import create_app

    reset_runtime()
    return TestClient(create_app())


class TestEdgarApi:
    def test_fetch_endpoint_indexes_filings(self, ai_env, tmp_path, monkeypatch):
        from src.ai import edgar as edgar_mod

        client = _make_client()

        def _fake(ingestor, cfg, ticker, *, forms=None, count=None, client=None):
            assert ticker == "NVDA"
            assert count == 2
            return EdgarFetchResult(ticker=ticker, cik=1045810, filings_found=1)

        monkeypatch.setattr(edgar_mod, "ingest_latest_filings", _fake)
        resp = client.post(
            "/ai/documents/edgar", json={"ticker": "nvda", "count": 2}
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["ticker"] == "NVDA"
        assert body["filings_found"] == 1

    def test_invalid_ticker_rejected(self, ai_env):
        resp = _make_client().post("/ai/documents/edgar", json={"ticker": "NV DA!"})
        assert resp.status_code == 422

    def test_requires_admin_token_when_configured(self, ai_env, monkeypatch):
        from src.ai import edgar as edgar_mod

        monkeypatch.setenv("ALPHAFORGE_AI_ADMIN_TOKEN", "topsecrettoken")
        # no live SEC traffic in tests - stub the fetch itself
        monkeypatch.setattr(
            edgar_mod,
            "ingest_latest_filings",
            lambda *a, **k: EdgarFetchResult(ticker="NVDA", cik=1045810, filings_found=0),
        )
        client = _make_client()
        assert client.post("/ai/documents/edgar", json={"ticker": "NVDA"}).status_code == 401
        ok = client.post(
            "/ai/documents/edgar",
            json={"ticker": "NVDA"},
            headers={"Authorization": "Bearer topsecrettoken"},
        )
        assert ok.status_code == 201
        assert ok.json()["ticker"] == "NVDA"

    def test_disabled_copilot_returns_503(self, ai_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "false")
        resp = _make_client().post("/ai/documents/edgar", json={"ticker": "NVDA"})
        assert resp.status_code == 503

    def test_disabled_edgar_returns_502_with_reason(self, ai_env, monkeypatch, tmp_path):
        monkeypatch.setenv("ALPHAFORGE_EDGAR_ENABLED", "false")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_CACHE_DIR", str(tmp_path / "edgar_cache"))
        resp = _make_client().post("/ai/documents/edgar", json={"ticker": "NVDA"})
        assert resp.status_code == 502
        assert "ALPHAFORGE_EDGAR_ENABLED" in resp.json()["detail"]


class TestFetchScript:
    def test_refuses_when_ai_disabled(self, monkeypatch, capsys):
        from scripts.fetch_edgar import main

        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "false")
        assert main(["NVDA"]) == 2
        assert "ALPHAFORGE_AI_ENABLED" in capsys.readouterr().err

    def test_refuses_when_edgar_disabled(self, monkeypatch, capsys):
        from scripts.fetch_edgar import main

        monkeypatch.setenv("ALPHAFORGE_AI_ENABLED", "true")
        monkeypatch.setenv("ALPHAFORGE_EDGAR_ENABLED", "false")
        assert main(["NVDA"]) == 2
        assert "ALPHAFORGE_EDGAR_ENABLED" in capsys.readouterr().err
