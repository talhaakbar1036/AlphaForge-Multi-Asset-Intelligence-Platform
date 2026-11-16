"""Leakage-safe features and windows for next-session direction models.

The contract is intentionally narrow:

* features at row ``t`` use only prices observed up to the close of ``t``;
* the label at row ``t`` compares ``close[t]`` with ``close[t-1]``;
* a window used to predict the label at ``t`` contains rows strictly before
  ``t``, so target-bar information can never enter its own input.

The six price-relative features match the schema of the pretrained artifacts
under ``artifacts/direction_lstm/``, so the same code serves training and
inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "close_return",
    "open_return",
    "high_return",
    "low_return",
    "intraday_range",
    "log_volume_change",
)

_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SequenceDataset:
    """Aligned model inputs, targets and the date/position of each target."""

    sequences: np.ndarray
    targets: np.ndarray
    target_dates: pd.DatetimeIndex
    target_positions: np.ndarray
    feature_frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class SequenceSplits:
    """Chronological train/validation/test partitions of a sequence dataset."""

    train: SequenceDataset
    validation: SequenceDataset
    test: SequenceDataset
    train_feature_end: int


def _canonical_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names and index without changing values."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("price data must be a non-empty DataFrame")

    frame = prices.copy()
    lower = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in _PRICE_COLUMNS if c not in lower]
    if missing:
        raise ValueError(f"price data is missing columns: {', '.join(missing)}")

    if "date" in lower:
        frame["date"] = pd.to_datetime(frame[lower["date"]], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()]

    rename = {lower[c]: c for c in _PRICE_COLUMNS}
    frame = frame.rename(columns=rename)
    for column in _PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.sort_index(kind="stable")
        .loc[~frame.index.duplicated(keep="last")]
        .dropna(subset=list(_PRICE_COLUMNS))
    )
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame.index.name = "Date"
    return frame


def build_direction_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features, labels)`` with a shared, unique DatetimeIndex.

    The first observation is dropped because returns require a prior close.
    Labels are float32 ``1.0`` for an up session and ``0.0`` otherwise.
    """
    frame = _canonical_prices(prices)
    if len(frame) < 3:
        raise ValueError("at least three valid price rows are required")

    previous_close = frame["close"].shift(1)
    features = pd.DataFrame(
        {
            "close_return": np.log(frame["close"] / previous_close),
            "open_return": np.log(frame["open"] / previous_close),
            "high_return": np.log(frame["high"] / previous_close),
            "low_return": np.log(frame["low"] / previous_close),
            "intraday_range": np.log(frame["high"] / frame["low"]),
            "log_volume_change": np.log1p(frame["volume"].clip(lower=0)).diff(),
        },
        index=frame.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan)

    labels = (frame["close"] > previous_close).astype("float32")
    valid = features.notna().all(axis=1) & previous_close.notna()
    features = features.loc[valid]
    labels = labels.loc[valid]
    if features.empty:
        raise ValueError("no valid feature rows were produced")
    return features, labels


def make_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    sequence_length: int,
) -> SequenceDataset:
    """Build windows ending immediately before each target row."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(features) != len(labels) or not features.index.equals(labels.index):
        raise ValueError("features and labels must share the same length and index")
    if len(features) <= sequence_length:
        raise ValueError(
            f"need more than {sequence_length} feature rows; found {len(features)}"
        )
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"features are missing columns: {', '.join(missing)}")

    values = features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = labels.to_numpy(dtype=np.float32)
    positions = np.arange(sequence_length, len(values))
    sequences = np.stack([values[pos - sequence_length : pos] for pos in positions])
    return SequenceDataset(
        sequences=sequences,
        targets=targets[positions],
        target_dates=features.index[positions],
        target_positions=positions,
        feature_frame=features,
    )


def _slice(dataset: SequenceDataset, indices: np.ndarray) -> SequenceDataset:
    return SequenceDataset(
        sequences=dataset.sequences[indices],
        targets=dataset.targets[indices],
        target_dates=dataset.target_dates[indices],
        target_positions=dataset.target_positions[indices],
        feature_frame=dataset.feature_frame,
    )


def split_sequences(
    dataset: SequenceDataset,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SequenceSplits:
    """Partition ordered windows chronologically (never shuffled)."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    n = len(dataset)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    if train_end == 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError(
            f"not enough sequences ({n}) for non-empty train, validation and test splits"
        )

    all_indices = np.arange(n)
    train = _slice(dataset, all_indices[:train_end])
    validation = _slice(dataset, all_indices[train_end:validation_end])
    test = _slice(dataset, all_indices[validation_end:])
    # Scaler fit rows: every input row used by a training window, and nothing
    # from the validation/test targets.
    train_feature_end = int(train.target_positions[-1])
    return SequenceSplits(
        train=train,
        validation=validation,
        test=test,
        train_feature_end=train_feature_end,
    )
