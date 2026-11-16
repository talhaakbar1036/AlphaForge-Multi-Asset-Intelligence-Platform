"""Load and serve the optional next-session direction LSTM.

TensorFlow is imported only when a prediction is requested, so the core API,
dashboard and test suite stay lightweight on installs without the ``ml``
extra.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.errors import (
    ModelArtifactNotFound,
    ModelDependencyError,
    ModelPredictionError,
)
from src.models.features import FEATURE_COLUMNS, build_direction_features
from src.utils.config import CONFIG
from src.utils.helpers import sanitize_symbol

DISCLAIMER = (
    "Research output only: a direction probability is not a price forecast, "
    "not investment advice, and is not blended into AlphaForge's rule-based "
    "decision or risk scores."
)


def _artifact_dir(artifacts_dir: Path | str | None) -> Path:
    return Path(artifacts_dir) if artifacts_dir else CONFIG.model_artifacts_dir


def _artifact_stem(symbol: str) -> str:
    return sanitize_symbol(symbol.upper())


def dependencies_available() -> dict[str, bool]:
    """Report optional ML dependency availability without importing them."""
    return {
        name: importlib.util.find_spec(module) is not None
        for name, module in (
            ("tensorflow", "tensorflow"),
            ("scikit-learn", "sklearn"),
            ("joblib", "joblib"),
        )
    }


def available_model_symbols(artifacts_dir: Path | str | None = None) -> list[str]:
    directory = _artifact_dir(artifacts_dir)
    if not directory.exists():
        return []
    suffix = "_direction_metrics.json"
    return sorted(
        path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}")
    )


def model_info(symbol: str, artifacts_dir: Path | str | None = None) -> dict[str, Any]:
    """Read report metadata for one symbol without importing TensorFlow."""
    directory = _artifact_dir(artifacts_dir)
    stem = _artifact_stem(symbol)
    metrics_path = directory / f"{stem}_direction_metrics.json"
    if not metrics_path.exists():
        raise ModelArtifactNotFound(
            f"no direction model for '{symbol}' under {directory}; "
            "train one with scripts/train_models.py or use a pretrained symbol"
        )
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPredictionError(f"cannot read model report {metrics_path}: {exc}") from exc


def model_catalog(artifacts_dir: Path | str | None = None) -> dict[str, Any]:
    """JSON-safe inventory used by ``GET /api/v1/models`` and the dashboard."""
    symbols = available_model_symbols(artifacts_dir)
    entries = []
    for symbol in symbols:
        try:
            metrics = model_info(symbol, artifacts_dir)
        except (ModelArtifactNotFound, ModelPredictionError):
            continue
        test = metrics.get("test", metrics)
        entries.append(
            {
                "symbol": symbol,
                "accuracy": test.get("accuracy", metrics.get("test_accuracy")),
                "majority_baseline_accuracy": test.get(
                    "majority_baseline_accuracy",
                    metrics.get("test_majority_class_baseline_accuracy"),
                ),
                "roc_auc": test.get("roc_auc", metrics.get("test_roc_auc")),
                "test_start": test.get(
                    "test_start", metrics.get("test_period_start")
                ),
                "test_end": test.get("test_end", metrics.get("test_period_end")),
                "trained_at": metrics.get("generated_at", metrics.get("trained_at")),
            }
        )
    dependencies = dependencies_available()
    return {
        "models": entries,
        "count": len(entries),
        "dependencies": dependencies,
        "ready": bool(entries) and all(dependencies.values()),
        "disclaimer": DISCLAIMER,
    }


class DirectionPredictor:
    """One symbol's scaler + Keras model pair, loaded on demand."""

    def __init__(
        self,
        symbol: str,
        model: Any,
        scaler: Any,
        *,
        feature_columns: tuple[str, ...],
        sequence_length: int,
        metrics: dict[str, Any],
        model_path: Path,
        scaler_path: Path,
    ) -> None:
        self.symbol = symbol
        self._model = model
        self._scaler = scaler
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.metrics = metrics
        self.model_path = model_path
        self.scaler_path = scaler_path

    @classmethod
    def load(
        cls,
        symbol: str,
        artifacts_dir: Path | str | None = None,
    ) -> "DirectionPredictor":
        directory = _artifact_dir(artifacts_dir)
        stem = _artifact_stem(symbol)
        model_path = directory / f"{stem}_direction_lstm.keras"
        scaler_path = directory / f"{stem}_direction_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise ModelArtifactNotFound(
                f"no trained artifacts for '{symbol}' under {directory}"
            )
        missing = [name for name, ok in dependencies_available().items() if not ok]
        if missing:
            raise ModelDependencyError(
                "missing optional ML dependencies: "
                + ", ".join(missing)
                + '; install with pip install -e ".[ml]"'
            )

        try:
            import joblib

            bundle = joblib.load(scaler_path)
        except Exception as exc:  # noqa: BLE001 - artifact corruption is data-dependent
            raise ModelPredictionError(f"cannot load scaler {scaler_path}: {exc}") from exc

        if not isinstance(bundle, dict) or "scaler" not in bundle:
            raise ModelPredictionError(
                f"unexpected scaler bundle schema in {scaler_path}"
            )
        features = tuple(bundle.get("features", FEATURE_COLUMNS))
        sequence_length = int(bundle.get("sequence_length", CONFIG.direction_model.sequence_length))

        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as exc:  # noqa: BLE001
            raise ModelDependencyError(
                f"cannot load TensorFlow model {model_path}: {exc}"
            ) from exc

        try:
            metrics = model_info(symbol, directory)
        except ModelArtifactNotFound:
            metrics = {}
        return cls(
            symbol=symbol,
            model=model,
            scaler=bundle["scaler"],
            feature_columns=features,
            sequence_length=sequence_length,
            metrics=metrics,
            model_path=model_path,
            scaler_path=scaler_path,
        )

    def predict(self, prices: pd.DataFrame) -> dict[str, Any]:
        """Predict the direction of the session following the last valid row."""
        try:
            features, _ = build_direction_features(prices)
        except ValueError as exc:
            raise ModelPredictionError(f"cannot build features for {self.symbol}: {exc}") from exc

        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ModelPredictionError(
                f"model expects columns missing from the data: {', '.join(missing)}"
            )
        if len(features) < self.sequence_length:
            raise ModelPredictionError(
                f"{self.symbol} has {len(features)} feature rows; "
                f"the model needs {self.sequence_length}"
            )

        window = features.loc[:, self.feature_columns].tail(self.sequence_length)
        values = window.to_numpy(dtype=np.float32).reshape(1, self.sequence_length, -1)
        try:
            scaled = self._scaler.transform(values.reshape(-1, values.shape[-1]))
            scaled = scaled.reshape(values.shape)
            raw = np.asarray(self._model.predict(scaled, verbose=0)).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - model/serving failures are surfaced
            raise ModelPredictionError(f"inference failed for {self.symbol}: {exc}") from exc
        if raw.size == 0 or not np.isfinite(raw[0]):
            raise ModelPredictionError(f"model returned an invalid probability for {self.symbol}")

        probability_up = float(np.clip(raw[0], 0.0, 1.0))
        threshold = float(CONFIG.direction_model.threshold)
        test = self.metrics.get("test", {})
        return {
            "symbol": self.symbol,
            "model": "direction_lstm",
            "prediction": "UP" if probability_up >= threshold else "DOWN",
            "probability_up": round(probability_up, 6),
            "probability_down": round(1.0 - probability_up, 6),
            "confidence": round(abs(probability_up - 0.5) * 2.0, 6),
            "threshold": threshold,
            "as_of": str(window.index[-1].date()),
            "predicted_for": "next trading session after as_of",
            "sequence_length": self.sequence_length,
            "features": list(self.feature_columns),
            "training_metrics": {
                "accuracy": test.get("accuracy", self.metrics.get("test_accuracy")),
                "majority_baseline_accuracy": test.get(
                    "majority_baseline_accuracy",
                    self.metrics.get("test_majority_class_baseline_accuracy"),
                ),
                "roc_auc": test.get("roc_auc", self.metrics.get("test_roc_auc")),
                "test_start": test.get("test_start", self.metrics.get("test_period_start")),
                "test_end": test.get("test_end", self.metrics.get("test_period_end")),
            },
            "disclaimer": DISCLAIMER,
        }
