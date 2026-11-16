"""Shared helpers: symbol naming and date inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import SYMBOL_ALIASES


def sanitize_symbol(symbol: str) -> str:
    """Yahoo symbol -> filesystem-safe stem (e.g. 'GC=F' -> 'GC_F')."""
    return SYMBOL_ALIASES.get(symbol, symbol.replace("=", "_").replace("^", ""))


def unsanitize_symbol(stem: str) -> str:
    """Best-effort inverse of :func:`sanitize_symbol` for display purposes."""
    for yahoo, safe in SYMBOL_ALIASES.items():
        if safe == stem:
            return yahoo
    return stem


def annualization_factor(index: pd.DatetimeIndex) -> int:
    """Periods per year inferred from median spacing (252 daily, 12 monthly)."""
    if len(index) < 3:
        return 252
    median_days = np.median(np.diff(index.values).astype("timedelta64[D]").astype(int))
    return 12 if median_days >= 20 else 252
