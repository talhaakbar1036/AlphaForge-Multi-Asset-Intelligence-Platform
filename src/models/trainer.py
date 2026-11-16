"""Train the optional Keras LSTM direction model without look-ahead leakage.

This is a deliberate, small training path: six price-relative features,
chronological 70/15/15 splits, a scaler fitted only on training inputs, and
early stopping. It is not an automated alpha factory, and the reported
baseline is kept next to accuracy on purpose.
"""
from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.errors import ModelDependencyError, ModelPredictionError
from src.models.features import (
    FEATURE_COLUMNS,
    build_direction_features,
    make_sequences,
    split_sequences,
)
from src.models.metrics import classification_metrics
from src.utils.config import CONFIG, DirectionModelConfig
from src.utils.helpers import sanitize_symbol
from src.utils.logger import get_logger

log = get_logger(__name__)


def _require_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        import joblib
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ModelDependencyError(
            'training requires TensorFlow, scikit-learn and joblib; '
            'install with pip install -e ".[ml]"'
        ) from exc
    return tf, StandardScaler, joblib


def _set_seeds(tf: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_lstm_model(tf: Any, cfg: DirectionModelConfig, n_features: int) -> Any:
    """Stacked LSTM + dropout + dense head, matching the published artifacts."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    layers: list[Any] = [
        tf.keras.layers.Input(shape=(cfg.sequence_length, n_features))
    ]
    for layer_index in range(cfg.num_layers):
        layers.append(
            tf.keras.layers.LSTM(
                cfg.hidden_units,
                return_sequences=layer_index < cfg.num_layers - 1,
                name=f"lstm_{layer_index + 1}",
            )
        )
        layers.append(
            tf.keras.layers.Dropout(cfg.dropout, name=f"dropout_{layer_index + 1}")
        )
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid", name="direction"))
    model = tf.keras.Sequential(layers, name="direction_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _scale_sequences(scaler: Any, sequences: np.ndarray) -> np.ndarray:
    flat = sequences.reshape(-1, sequences.shape[-1])
    return scaler.transform(flat).reshape(sequences.shape).astype(np.float32)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def train_symbol(
    symbol: str,
    prices: pd.DataFrame,
    *,
    cfg: DirectionModelConfig | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train, evaluate and persist one symbol's direction model."""
    cfg = cfg or CONFIG.direction_model
    cfg.validate()
    tf, StandardScaler, joblib = _require_training_dependencies()
    _set_seeds(tf, cfg.seed)

    features, labels = build_direction_features(prices)
    dataset = make_sequences(features, labels, cfg.sequence_length)
    splits = split_sequences(
        dataset,
        cfg.train_ratio,
        cfg.validation_ratio,
        cfg.test_ratio,
    )

    fit_rows = features.loc[:, FEATURE_COLUMNS].iloc[: splits.train_feature_end]
    if len(fit_rows) < cfg.sequence_length:
        raise ModelPredictionError(
            f"not enough training rows for {symbol}: {len(fit_rows)}"
        )
    scaler = StandardScaler()
    scaler.fit(fit_rows.to_numpy(dtype=np.float32))

    train_x = _scale_sequences(scaler, splits.train.sequences)
    validation_x = _scale_sequences(scaler, splits.validation.sequences)
    test_x = _scale_sequences(scaler, splits.test.sequences)
    train_y = splits.train.targets.astype(np.float32)
    validation_y = splits.validation.targets.astype(np.float32)
    test_y = splits.test.targets.astype(np.float32)

    model = build_lstm_model(tf, cfg, train_x.shape[-1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    probabilities = np.asarray(model.predict(test_x, verbose=0)).reshape(-1)
    metrics = classification_metrics(test_y, probabilities, threshold=cfg.threshold)

    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "symbol": symbol,
        "model": "direction_lstm",
        "generated_at": generated_at,
        "sequence_length": cfg.sequence_length,
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "rows_used": int(len(features)),
        "sequences": int(len(dataset)),
        "train_sequences": int(len(splits.train)),
        "validation_sequences": int(len(splits.validation)),
        "test_sequences": int(len(splits.test)),
        "train_period_start": str(splits.train.target_dates[0].date()),
        "train_period_end": str(splits.train.target_dates[-1].date()),
        "validation_period_start": str(splits.validation.target_dates[0].date()),
        "validation_period_end": str(splits.validation.target_dates[-1].date()),
        "test_period_start": str(splits.test.target_dates[0].date()),
        "test_period_end": str(splits.test.target_dates[-1].date()),
        "test_accuracy": metrics["accuracy"],
        "test_majority_class_baseline_accuracy": metrics["majority_baseline_accuracy"],
        "test_roc_auc": metrics["roc_auc"],
        "test": metrics,
        "epochs_requested": cfg.epochs,
        "epochs_completed": int(len(history.history.get("loss", []))),
        "seed": cfg.seed,
        "threshold": cfg.threshold,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "source": "alphaforge",
    }

    directory = Path(output_dir) if output_dir else CONFIG.model_artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = sanitize_symbol(symbol.upper())
    with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=directory) as temp:
        temp_path = Path(temp)
        model.save(temp_path / "model.keras")
        joblib.dump(
            {
                "scaler": scaler,
                "features": list(FEATURE_COLUMNS),
                "sequence_length": cfg.sequence_length,
                "symbol": symbol,
            },
            temp_path / "scaler.joblib",
        )
        _atomic_write_json(temp_path / "metrics.json", report)
        (temp_path / "model.keras").replace(directory / f"{stem}_direction_lstm.keras")
        (temp_path / "scaler.joblib").replace(directory / f"{stem}_direction_scaler.joblib")
        (temp_path / "metrics.json").replace(directory / f"{stem}_direction_metrics.json")

    log.info(
        "trained %s: accuracy %.4f vs baseline %.4f (AUC %s)",
        symbol,
        metrics["accuracy"],
        metrics["majority_baseline_accuracy"],
        metrics["roc_auc"],
    )
    return report


def training_summary(
    reports: list[dict[str, Any]],
    failures: dict[str, str],
    cfg: DirectionModelConfig,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "output_dir": str(output_dir),
        "trained": reports,
        "failed": failures,
        "warning": (
            "Direction accuracy is compared with the majority-class baseline. "
            "Near-baseline results are expected and must not be presented as "
            "a profitable trading edge."
        ),
    }
