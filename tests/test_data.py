"""Tests for the data layer: loading, cleaning, alignment, features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data_processor import align_macro_daily, clean_ohlcv, detect_outliers
from src.data.feature_engineering import make_targets, technical_indicators
from src.utils.helpers import sanitize_symbol


def _fake_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, n),
            "High": close + np.abs(rng.normal(0.5, 0.2, n)),
            "Low": close - np.abs(rng.normal(0.5, 0.2, n)),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def test_sanitize_symbols():
    assert sanitize_symbol("GC=F") == "GC_F"
    assert sanitize_symbol("^GSPC") == "GSPC"
    assert sanitize_symbol("AAPL") == "AAPL"


def test_clean_ohlcv_dedupes_and_fills():
    df = _fake_ohlcv(100)
    dirty = pd.concat([df, df.iloc[[-1]]])  # duplicate last row
    cleaned = clean_ohlcv(dirty)
    assert cleaned.index.is_unique
    assert not cleaned["Close"].isna().any()


def test_outlier_detection_flags_crash():
    df = _fake_ohlcv(200)
    df.iloc[150, df.columns.get_loc("Close")] *= 0.5  # synthetic crash
    flagged = detect_outliers(df)
    assert bool(flagged["outlier"].iloc[151]) or bool(flagged["outlier"].iloc[150])


def test_macro_alignment_no_leakage():
    monthly = pd.DataFrame({"cpi": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    daily_idx = pd.bdate_range("2024-01-01", "2024-03-15")
    daily = align_macro_daily(monthly, daily_idx)
    # January's value must only appear after February starts (shift(1))
    jan_days = daily.loc["2024-01-01":"2024-01-31", "cpi"]
    assert jan_days.isna().all()
    feb_first = daily.loc["2024-02-01", "cpi"]
    assert feb_first == 100.0


def test_technical_indicators_columns():
    feats = technical_indicators(_fake_ohlcv())
    for col in ("rsi_14", "macd", "bb_width", "atr_14", "close_vs_sma_50", "vol_z", "realized_vol_21"):
        assert col in feats.columns
    assert feats["rsi_14"].dropna().between(0, 100).all()


def test_targets_are_shifted():
    feats = technical_indicators(_fake_ohlcv())
    out = make_targets(feats)
    expected = out["Close"].shift(-1) / out["Close"] - 1.0
    pd.testing.assert_series_equal(out["next_day_return"], expected, check_names=False)
    assert (out["direction"].dropna().isin([0.0, 1.0])).all()
