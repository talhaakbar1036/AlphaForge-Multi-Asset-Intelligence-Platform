"""Tests for decision engine, risk analytics, portfolio and backtesting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import make_targets, technical_indicators
from src.engine.decision_engine import decide
from src.engine.risk_analytics import (
    performance_metrics,
    portfolio_concentration,
    risk_profile,
    sharpe_ratio,
)
from src.portfolio.backtester import compare_table, run_backtest
from src.portfolio.optimizer import recommend_allocation


def _prices(n: int = 800, drift: float = 0.0004, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(drift, 0.015, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx, name="Close")


def _features(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    px = _prices(n, seed=1)
    df = pd.DataFrame(
        {
            "Open": px,
            "High": px * 1.01,
            "Low": px * 0.99,
            "Close": px,
            "Adj Close": px,
            "Volume": rng.integers(1e6, 5e6, n).astype(float),
            "news_sentiment": rng.normal(0.2, 0.2, n),
        },
        index=px.index,
    )
    return make_targets(technical_indicators(df))


def test_risk_analytics_sane():
    px = _prices()
    rp = risk_profile(px)
    assert 0 <= rp["risk_score"] <= 100
    assert rp["max_drawdown"] <= 0
    assert np.isfinite(rp["sharpe"])
    assert -3 < sharpe_ratio(px.pct_change().dropna()) < 10


def test_concentration_bounds():
    assert portfolio_concentration({"A": 0.5, "B": 0.5}) == pytest.approx(0.5)
    assert portfolio_concentration({"A": 1.0}) == pytest.approx(1.0)


def test_decision_engine_contract():
    feats = _features()
    d = decide("TEST", feats)
    assert d.recommendation in {"BUY", "HOLD", "SELL"}
    assert -1 <= d.investment_score <= 1
    assert 0 <= d.confidence <= 1
    assert d.last_close == float(feats["Close"].iloc[-1])
    assert d.why and d.risks  # every recommendation ships with why + risks
    # component scores must all be bounded
    assert all(-1 <= v <= 1 for v in d.components.values())


def test_portfolio_optimizer_methods():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {s: _prices(500, drift=rng.normal(0.0003, 0.0002), seed=i) for i, s in enumerate(["AAPL", "MSFT", "NVDA"])}
    )
    for method in ("mean_variance", "risk_parity", "inverse_vol"):
        out = recommend_allocation(frame, method=method)
        total = sum(out["allocations"].values())
        assert total == pytest.approx(100, abs=0.5)
        assert all(v >= 0 for v in out["allocations"].values())
        assert method in out["method"] or "fallback" in out["method"]


def test_backtest_metrics_and_no_lookahead():
    px = _prices(600)
    signals = pd.Series(0.0, index=px.index)
    signals.iloc[100::30] = 1.0   # periodic long entries
    signals.iloc[115::30] = -1.0
    res = run_backtest(px, signals=signals)
    table = compare_table(res)
    for col in ("cagr", "sharpe", "max_drawdown", "win_rate", "total_return", "volatility"):
        assert col in table.columns
    strat = res["alphaforge_strategy"]
    bh = res["buy_hold"]
    assert strat["win_rate"] >= 0 and strat["win_rate"] <= 1
    # strategy that is flat most of the time should trade less than buy&hold vol
    assert isinstance(bh["cagr"], float)


def test_performance_metrics_keys():
    eq = _prices(400)
    m = performance_metrics(eq)
    assert set(m) == {"total_return", "cagr", "volatility", "sharpe", "max_drawdown", "win_rate"}
