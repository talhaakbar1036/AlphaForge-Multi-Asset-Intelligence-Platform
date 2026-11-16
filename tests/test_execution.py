"""Opt-in trading execution: interlocks, guardrails, broker, audit trail.

Everything here runs offline: the paper broker fills at monkeypatched local
prices and the Alpaca adapter is exercised against a mocked HTTP layer.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
import requests

from src.execution.broker import AlpacaBroker, PaperBroker
from src.execution.config import TradingConfig, TradingConfigError, reset_trading_config
from src.execution.errors import (
    BrokerError,
    GuardrailError,
    OrderNotFound,
    TradingDisabledError,
)
from src.execution.models import OrderRequest
from src.execution.service import ExecutionService, reset_execution_service

SECRET = "s3cr3t-alpaca-key-value"


def _table(price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame({"Close": [price, price, price]}, index=idx)


@pytest.fixture
def exec_env(tmp_path, monkeypatch):
    """Enabled dry-run configuration on tmp paths, secrets stripped."""
    monkeypatch.setenv("ALPHAFORGE_TRADING_ENABLED", "true")
    monkeypatch.setenv("ALPHAFORGE_TRADING_MODE", "dry_run")
    monkeypatch.setenv("ALPHAFORGE_TRADING_BROKER", "paper")
    monkeypatch.setenv("ALPHAFORGE_TRADING_ALLOW_LIVE", "false")
    monkeypatch.setenv("ALPHAFORGE_TRADING_ALLOW_SHORT", "false")
    monkeypatch.setenv("ALPHAFORGE_TRADING_UNIVERSE", "NVDA,AAPL,MSFT")
    monkeypatch.setenv("ALPHAFORGE_TRADING_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ALPHAFORGE_TRADING_PAPER_STATE", str(tmp_path / "paper.json"))
    monkeypatch.delenv("ALPHAFORGE_TRADING_ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPHAFORGE_TRADING_ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPHAFORGE_TRADING_KILL_SWITCH", raising=False)
    reset_trading_config()
    reset_execution_service()
    yield
    reset_trading_config()
    reset_execution_service()


def _service(env) -> ExecutionService:
    from src.execution.config import get_trading_config

    return ExecutionService(get_trading_config())


def _audit_lines(cfg) -> list[dict]:
    if not cfg.audit_path.exists():
        return []
    return [json.loads(line) for line in cfg.audit_path.read_text().splitlines() if line.strip()]


class TestConfigInterlocks:
    def test_disabled_by_default(self):
        cfg = TradingConfig()
        assert cfg.enabled is False
        assert cfg.mode == "dry_run"
        cfg.validate()

    def test_invalid_mode_rejected(self):
        with pytest.raises(TradingConfigError, match="ALPHAFORGE_TRADING_MODE"):
            replace(TradingConfig(), mode="yolo").validate()

    def test_invalid_broker_rejected(self):
        with pytest.raises(TradingConfigError, match="ALPHAFORGE_TRADING_BROKER"):
            replace(TradingConfig(), broker=" robinhood ").validate()

    def test_live_requires_allow_live_switch(self):
        cfg = replace(
            TradingConfig(),
            enabled=True,
            mode="live",
            broker="alpaca",
            allow_live=False,
            alpaca_key_id="k",
            alpaca_secret_key=SECRET,
            alpaca_base_url="https://api.alpaca.markets",
        )
        with pytest.raises(TradingConfigError, match="ALLOW_LIVE"):
            cfg.validate()

    def test_live_requires_alpaca_credentials(self):
        cfg = replace(
            TradingConfig(),
            enabled=True,
            mode="live",
            broker="alpaca",
            allow_live=True,
            alpaca_base_url="https://api.alpaca.markets",
        )
        with pytest.raises(TradingConfigError, match="ALPACA_KEY_ID"):
            cfg.validate()

    def test_live_refuses_paper_base_url(self):
        cfg = replace(
            TradingConfig(),
            enabled=True,
            mode="live",
            broker="alpaca",
            allow_live=True,
            alpaca_key_id="k",
            alpaca_secret_key=SECRET,
        )  # alpaca_base_url defaults to the paper URL
        with pytest.raises(TradingConfigError, match="paper URL"):
            cfg.validate()

    def test_live_rejects_paper_broker(self):
        cfg = replace(
            TradingConfig(),
            enabled=True,
            mode="live",
            broker="paper",
            allow_live=True,
        )
        with pytest.raises(TradingConfigError, match="ALPHAFORGE_TRADING_BROKER=alpaca"):
            cfg.validate()

    def test_bad_caps_rejected(self):
        with pytest.raises(TradingConfigError, match="MAX_ORDER_NOTIONAL"):
            replace(TradingConfig(), max_order_notional=0).validate()
        with pytest.raises(TradingConfigError, match="MAX_OPEN_ORDERS"):
            replace(TradingConfig(), max_open_orders=0).validate()

    def test_public_dict_never_contains_secrets(self, exec_env, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_TRADING_ALPACA_KEY_ID", "AKID-canary")
        monkeypatch.setenv("ALPHAFORGE_TRADING_ALPACA_SECRET_KEY", SECRET)
        reset_trading_config()
        from src.execution.config import get_trading_config

        cfg = get_trading_config()
        blob = json.dumps(cfg.as_public_dict())
        assert "AKID-canary" not in blob
        assert SECRET not in blob
        assert cfg.as_public_dict()["alpaca_key_configured"] is True


class TestGuardrails:
    def test_disabled_service_refuses_orders(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_TRADING_ENABLED", "false")
        monkeypatch.setenv("ALPHAFORGE_TRADING_AUDIT_PATH", str(tmp_path / "a.jsonl"))
        monkeypatch.setenv("ALPHAFORGE_TRADING_PAPER_STATE", str(tmp_path / "p.json"))
        reset_trading_config()
        service = _service(None)
        with pytest.raises(TradingDisabledError, match="ALPHAFORGE_TRADING_ENABLED"):
            service.submit_order(OrderRequest(symbol="NVDA", side="buy", qty=1))

    def test_kill_switch_halts_everything(self, exec_env):
        from src.execution.config import get_trading_config

        cfg = replace(get_trading_config(), kill_switch=True)
        service = ExecutionService(cfg)
        with pytest.raises(TradingDisabledError, match="kill switch"):
            service.submit_order(OrderRequest(symbol="NVDA", side="buy", qty=1))

    def test_unlisted_symbol_rejected_and_audited(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        service = _service(None)
        with pytest.raises(GuardrailError, match="allow-list"):
            service.submit_order(OrderRequest(symbol="TSLA", side="buy", qty=1))
        actions = [line["action"] for line in _audit_lines(service.cfg)]
        assert "order_rejected" in actions

    def test_notional_cap(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        service = _service(None)
        with pytest.raises(GuardrailError, match="exceeds the per-order cap"):
            service.submit_order(OrderRequest(symbol="NVDA", side="buy", qty=10))  # $1000 > $500

    def test_short_selling_blocked_by_default(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        service = _service(None)
        with pytest.raises(GuardrailError, match="short selling is disabled"):
            service.submit_order(OrderRequest(symbol="NVDA", side="sell", qty=1))

    def test_order_requires_sizing(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        service = _service(None)
        with pytest.raises(GuardrailError, match="either qty or notional"):
            service.submit_order(OrderRequest(symbol="NVDA", side="buy"))

    def test_open_order_cap(self, exec_env, monkeypatch):
        from src.execution.config import get_trading_config
        from src.execution.models import Position

        class StubBroker:
            """Returns unfilled ('submitted') orders so the cap can trip."""

            name = "stub"

            def submit(self, request):
                from src.execution.models import OrderRecord

                return OrderRecord(
                    client_order_id=request.client_order_id,
                    broker_order_id="stub-1",
                    symbol=request.symbol,
                    side=request.side,
                    qty=request.qty,
                    notional=request.notional,
                    order_type=request.order_type,
                    limit_price=request.limit_price,
                    status="submitted",
                    mode="paper",
                    reason=request.reason,
                )

            def get_positions(self):
                return [Position("NVDA", 1.0, 100.0, 100.0)]

            def get_cash(self):
                return 0.0

            def list_orders(self):
                return []

            def get_order(self, client_order_id):
                return None

        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        # paper mode routes through the broker, so unfilled orders stay "open"
        cfg = replace(get_trading_config(), max_open_orders=1, mode="paper")
        service = ExecutionService(cfg, broker=StubBroker())
        service.submit_order(OrderRequest(symbol="NVDA", side="buy", qty=1))
        with pytest.raises(GuardrailError, match="open-order cap"):
            service.submit_order(OrderRequest(symbol="MSFT", side="buy", qty=1))


class TestDryRunFlow:
    def test_records_order_without_broker_call(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        service = _service(None)
        record = service.submit_order(
            OrderRequest(symbol="NVDA", side="buy", qty=2, reason="unit test")
        )
        assert record.status == "dry_run"
        assert record.mode == "dry_run"
        assert record.notional == 200.0
        actions = [line["action"] for line in _audit_lines(service.cfg)]
        assert "order_accepted" in actions

    def test_signal_flow_places_buy_and_ignores_hold(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        monkeypatch.setattr(
            "src.api.service.decision_payload",
            lambda s: {
                "recommendation": "BUY",
                "investment_score": 0.8,
                "confidence": 0.7,
                "as_of": "2024-01-03",
            },
        )
        service = _service(None)
        records = service.orders_from_signal("NVDA")
        assert len(records) == 1
        assert records[0].side == "buy"
        assert "engine signal BUY" in records[0].reason

        monkeypatch.setattr(
            "src.api.service.decision_payload",
            lambda s: {"recommendation": "HOLD", "investment_score": 0.01},
        )
        assert service.orders_from_signal("AAPL") == []
        actions = [line["action"] for line in _audit_lines(service.cfg)]
        assert "signal_ignored" in actions

    def test_status_is_secret_free(self, exec_env):
        status = _service(None).status()
        assert status["config"]["enabled"] is True
        assert status["config"]["mode"] == "dry_run"
        assert "not investment advice" in status["disclaimer"].lower() or "advice" in status["disclaimer"].lower()

    def test_unknown_order_lookup(self, exec_env):
        service = _service(None)
        with pytest.raises(OrderNotFound):
            service.get_order("nope")


class TestPaperBroker:
    def test_fill_persists_across_instances(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=50.0))
        from src.execution.config import get_trading_config

        cfg = get_trading_config()
        broker = PaperBroker(cfg.paper_state_path, starting_cash=1000.0)
        record = broker.submit(OrderRequest(symbol="NVDA", side="buy", qty=2))
        assert record.status == "filled"
        assert record.filled_price == 50.0
        assert broker.get_cash() == 900.0

        reloaded = PaperBroker(cfg.paper_state_path, starting_cash=1000.0)
        assert reloaded.get_cash() == 900.0
        positions = {p.symbol: p for p in reloaded.get_positions()}
        assert positions["NVDA"].qty == 2
        assert positions["NVDA"].market_value == 100.0

    def test_insufficient_cash_rejected(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        from src.execution.config import get_trading_config

        broker = PaperBroker(get_trading_config().paper_state_path, starting_cash=10.0)
        with pytest.raises(BrokerError, match="insufficient paper cash"):
            broker.submit(OrderRequest(symbol="NVDA", side="buy", qty=5))

    def test_sell_without_position_rejected(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        from src.execution.config import get_trading_config

        broker = PaperBroker(get_trading_config().paper_state_path)
        with pytest.raises(BrokerError, match="no paper position"):
            broker.submit(OrderRequest(symbol="NVDA", side="sell", qty=1))

    def test_no_price_data_is_a_typed_error(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: None)
        from src.execution.config import get_trading_config

        broker = PaperBroker(get_trading_config().paper_state_path)
        with pytest.raises(BrokerError, match="no local price data"):
            broker.submit(OrderRequest(symbol="NVDA", side="buy", qty=1))


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else ""
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestAlpacaBroker:
    def _broker(self) -> AlpacaBroker:
        return AlpacaBroker("AKID", SECRET, "https://paper-api.alpaca.markets")

    def test_missing_credentials_rejected(self):
        with pytest.raises(BrokerError, match="credentials are not configured"):
            AlpacaBroker("", "", "https://paper-api.alpaca.markets")

    def test_submit_builds_order_payload(self, monkeypatch):
        captured = {}

        def fake_request(method, url, headers=None, json=None, timeout=None):
            captured.update(method=method, url=url, headers=headers, payload=json)
            return _Resp(200, {"id": "ord-1", "symbol": "NVDA", "side": "buy", "status": "accepted"})

        monkeypatch.setattr("src.execution.broker.requests.request", fake_request)
        record = self._broker().submit(
            OrderRequest(symbol="NVDA", side="buy", qty=1, reason="r")
        )
        assert record.broker_order_id == "ord-1"
        assert captured["url"].endswith("/v2/orders")
        assert captured["headers"]["APCA-API-KEY-ID"] == "AKID"
        assert captured["payload"]["client_order_id"] == record.client_order_id

    def test_broker_error_maps_to_typed_error(self, monkeypatch):
        monkeypatch.setattr(
            "src.execution.broker.requests.request",
            lambda *a, **k: _Resp(422, {"message": "invalid order"}),
        )
        with pytest.raises(BrokerError, match="422"):
            self._broker().submit(OrderRequest(symbol="NVDA", side="buy", qty=1))

    def test_timeout_maps_to_typed_error(self, monkeypatch):
        def boom(*a, **k):
            raise requests.Timeout("slow")

        monkeypatch.setattr("src.execution.broker.requests.request", boom)
        with pytest.raises(BrokerError, match="timed out"):
            self._broker().get_cash()


class TestExecutionApi:
    def test_status_reports_disabled_without_secrets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_TRADING_ENABLED", "false")
        monkeypatch.setenv("ALPHAFORGE_TRADING_ALPACA_SECRET_KEY", SECRET)
        monkeypatch.setenv("ALPHAFORGE_TRADING_AUDIT_PATH", str(tmp_path / "a.jsonl"))
        monkeypatch.setenv("ALPHAFORGE_TRADING_PAPER_STATE", str(tmp_path / "p.json"))
        reset_trading_config()
        reset_execution_service()
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        client = TestClient(create_app())
        resp = client.get("/execution/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"]["enabled"] is False
        assert SECRET not in resp.text

    def test_orders_refused_while_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_TRADING_ENABLED", "false")
        monkeypatch.setenv("ALPHAFORGE_TRADING_AUDIT_PATH", str(tmp_path / "a.jsonl"))
        monkeypatch.setenv("ALPHAFORGE_TRADING_PAPER_STATE", str(tmp_path / "p.json"))
        reset_trading_config()
        reset_execution_service()
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        client = TestClient(create_app())
        assert client.get("/execution/orders").status_code == 503
        resp = client.post("/execution/orders", json={"symbol": "NVDA", "side": "buy", "qty": 1})
        assert resp.status_code == 503
        assert "ALPHAFORGE_TRADING_ENABLED" in resp.json()["detail"]

    def test_order_round_trip_in_dry_run(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        client = TestClient(create_app())
        resp = client.post(
            "/execution/orders",
            json={"symbol": "NVDA", "side": "buy", "notional": 250.0, "reason": "test"},
        )
        assert resp.status_code == 201
        order = resp.json()["order"]
        assert order["status"] == "dry_run"

        listed = client.get("/execution/orders")
        assert listed.status_code == 200
        assert any(o["client_order_id"] == order["client_order_id"] for o in listed.json()["orders"])

        single = client.get(f"/execution/orders/{order['client_order_id']}")
        assert single.status_code == 200
        assert single.json()["symbol"] == "NVDA"

        missing = client.get("/execution/orders/does-not-exist")
        assert missing.status_code == 404

    def test_rejected_order_returns_400(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table(price=100.0))
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        client = TestClient(create_app())
        resp = client.post("/execution/orders", json={"symbol": "TSLA", "side": "buy", "qty": 1})
        assert resp.status_code == 400
        assert "allow-list" in resp.json()["detail"]

    def test_signal_endpoint(self, exec_env, monkeypatch):
        monkeypatch.setattr("src.api.service.load_feature_table", lambda s: _table())
        monkeypatch.setattr(
            "src.api.service.decision_payload",
            lambda s: {"recommendation": "BUY", "investment_score": 0.9, "confidence": 0.8},
        )
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        client = TestClient(create_app())
        resp = client.post("/execution/signal", json={"symbol": "NVDA"})
        assert resp.status_code == 200
        assert resp.json()["placed"] == 1

    def test_positions_empty_in_dry_run(self, exec_env):
        from fastapi.testclient import TestClient
        from src.api.fastapi_app import create_app

        resp = TestClient(create_app()).get("/execution/positions")
        assert resp.status_code == 200
        assert resp.json()["positions"] == []
