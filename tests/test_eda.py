"""Tests for the EDA module (synthetic data - no disk access)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.eda import (
    _adf_pvalue,
    cross_asset_correlation,
    overview_table,
    sentiment_vs_returns,
    volatility_clustering_stats,
)


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n)), index=idx)
    return pd.DataFrame({"Close": px, "Volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def _universe() -> dict[str, dict[str, pd.DataFrame]]:
    return {"stocks": {s: _frame(seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}}


def test_overview_table_columns_and_values():
    table = overview_table(_universe())
    assert list(table.columns)[:4] == ["group", "symbol", "rows", "start"]
    assert len(table) == 3
    assert (table["rows"] == 600).all()
    assert (table["missing_pct"] == 0.0).all()


def test_adf_pvalue_stationary_vs_trending():
    rng = np.random.default_rng(1)
    white_noise = rng.normal(0, 0.01, 1000)
    trend = np.cumsum(rng.normal(0.01, 0.01, 1000))
    assert _adf_pvalue(white_noise) <= 0.05
    assert _adf_pvalue(trend) >= 0.10


def test_volatility_clustering_shape():
    stats = volatility_clustering_stats(_universe())
    assert stats.shape[0] == 3
    assert "acf_absret_lag1" in stats.columns
    assert stats["acf_absret_lag1"].between(-1, 1).all()


def test_cross_asset_correlation_bounds():
    uni = _universe()
    uni["etfs"] = {"ETF1": _frame(seed=9)}
    corr, fig = cross_asset_correlation(uni)
    assert corr.shape == (4, 4)
    assert ((corr >= -1) & (corr <= 1)).fillna(True).all().all()


def test_sentiment_bucketing():
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(2)
    feats = pd.DataFrame(
        {
            "Close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
            "news_sentiment": rng.choice([-0.6, 0.0, 0.6], 300),
        },
        index=idx,
    )
    table, fig = sentiment_vs_returns("TEST", features=feats)
    assert not table.empty
    assert set(table.index.astype(str)) == {"negative", "neutral", "positive"}
