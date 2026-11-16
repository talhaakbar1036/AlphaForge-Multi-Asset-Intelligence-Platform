"""Resolve a symbol to the best locally available OHLCV history.

Processed feature tables are preferred because they are the pipeline's
canonical, cleaned artifacts. Raw CSVs are a fallback for training a symbol
that has not gone through feature engineering yet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import CONFIG
from src.utils.helpers import sanitize_symbol


def load_price_history(symbol: str, *, data_dir: Path | None = None) -> pd.DataFrame:
    """Load cleaned OHLCV for ``symbol`` from processed or raw artifacts."""
    stem = sanitize_symbol(symbol)
    processed_dir = data_dir or CONFIG.data_processed
    processed = processed_dir / f"{stem}_features.parquet"
    if processed.exists():
        frame = pd.read_parquet(processed)
        if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
            return frame

    from src.data.data_processor import load_raw_price

    return load_raw_price(symbol)


def price_history_source(symbol: str, *, data_dir: Path | None = None) -> str:
    stem = sanitize_symbol(symbol)
    processed = (data_dir or CONFIG.data_processed) / f"{stem}_features.parquet"
    return "processed_features" if processed.exists() else "raw_csv"
