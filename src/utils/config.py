"""Central configuration for AlphaForge AI.

Single source of truth for paths, universes (tickers), date ranges and
decision-engine weights. Everything downstream imports from here so a source
or universe change never requires touching pipeline code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- universe --
TICKERS: dict[str, list[str]] = {
    "stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD"],
    "indices": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"],
    "etfs": ["SPY", "QQQ", "XLK", "SOXX", "VGT"],
    "commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "forex": ["EURUSD=X", "GBPUSD=X", "JPY=X", "CNY=X"],
}

# Yahoo symbol -> sanitized file stem used on disk (data/raw/<group>/<stem>.csv)
SYMBOL_ALIASES: dict[str, str] = {
    "^GSPC": "GSPC",
    "^IXIC": "IXIC",
    "^DJI": "DJI",
    "^RUT": "RUT",
    "^VIX": "VIX",
    "GC=F": "GC_F",
    "SI=F": "SI_F",
    "CL=F": "CL_F",
    "NG=F": "NG_F",
    "EURUSD=X": "EURUSD_X",
    "GBPUSD=X": "GBPUSD_X",
    "JPY=X": "JPY_X",
    "CNY=X": "USDCNY_X",
}

FRED_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "unemployment": "UNRATE",
    "ppi": "PPIACO",
    "dollar_index": "DTWEXBGS",
}

# ------------------------------------------------------------------- dates --
DATA_START = "2015-01-01"


@dataclass
class DirectionModelConfig:
    """Hyperparameters and split policy for the optional direction LSTM.

    Kept separate from :class:`DecisionConfig` on purpose: the ML signal is a
    research output and is *not* blended into the rule-based decision engine.
    """

    sequence_length: int = 60
    hidden_units: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.025
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    threshold: float = 0.5
    seed: int = 42

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.hidden_units <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_units and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, epochs and patience must be positive")
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train/validation/test ratios must sum to 1.0")


@dataclass
class DecisionConfig:
    """Weights of the investment decision engine (sum to 1.0)."""

    w_technical: float = 0.35    # RSI/MACD/MA composite
    w_trend: float = 0.25        # price vs SMA50/200 regime
    w_volatility: float = 0.15   # volatility penalty (negative weight)
    w_macro: float = 0.15        # macro regime score
    w_sentiment: float = 0.10    # news sentiment
    buy_threshold: float = 0.15
    sell_threshold: float = -0.15


@dataclass
class Config:
    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    model_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "direction_lstm"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "figures"

    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))

    decision: DecisionConfig = field(default_factory=DecisionConfig)
    direction_model: DirectionModelConfig = field(default_factory=DirectionModelConfig)

    def ensure_dirs(self) -> None:
        for p in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.reports_dir,
            self.figures_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
